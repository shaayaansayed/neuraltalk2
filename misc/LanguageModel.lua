require 'nn'
require 'cutorch'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'
local Attention = require 'misc.Attention'

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)

  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.seq_per_img = utils.getopt(opt, 'seq_per_img')
  self.batch_size = utils.getopt(opt, 'batch_size')
  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, 512, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  -- self.att = Attention.att(self.batch_size, self.seq_per_img, 196, 512, 512, self.rnn_size, dropout)
  self.att = Attention.att(self.batch_size*self.seq_per_img, 196, 512, self.rnn_size, self.input_encoding_size, dropout)
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  self.att_clones = {self.att}
  for t=2,self.seq_length+1 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
    self.att_clones[t] = self.att:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table, self.att}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()
  local p3,g3 = self.att:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.att_clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.att_clones) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(avg_context, feats, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(avg_context, feats, opt) end -- indirection for beam search

  local batch_size = avg_context:size(1)
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step

  for t=1,self.seq_length+1 do

    local xt, it, context, sampleLogprobs
    if t == 1 then
      context = avg_context
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- for _,node in ipairs(self.att.forwardnodes) do
      --   print('cool')
      --   if node.data.annotations.name == "scores" then
      --     print(node.data.module)
      --     -- print(node.data.module.output)
      --   end
      -- end

      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
      context = self.att:forward{feats, state[self.num_state], xt}
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    local inputs = {xt,context,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

function layer:sample_beam(avg_context, feats, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, context_size = feats:size(1), avg_context:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream
  print('sample beam...')

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  print(batch_size)
  for k=1,batch_size do
    print(k)
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+1 do

      local xt, it, sampleLogprobs
      local new_state
      local context
      if t == 1 then
        -- feed in the images
        context = avg_context[{ {k,k} }]:expand(beam_size, context_size) 

        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
        local all_context = self.att:forward{feats, state[self.num_state]}
        context = all_context[{{k,k}}]:expand(beam_size, context_size)

        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 2 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        it = beam_seq[t-1]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt,context,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

function layer:updateOutput(input)
  local seq = input[3]
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  self.att_inputs = {}
  for t=1,self.seq_length+1 do
    local can_skip = false
    local xt
    local context
    if t == 1 then
      -- feed in the start tokens
      context = input[1]
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)

        local feats = input[2]
        self.att_inputs[t] = {feats,self.state[t-1][#self.state[t-1]], xt}
        context = self.att_clones[t]:forward(self.att_inputs[t])
      end
    end

    if not can_skip then
      -- construct the inputs
      self.inputs[t] = {xt,context,unpack(self.state[t-1])}
      local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dimgs -- grad on input images

  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    dstate[t-1] = {} -- copy over rest to state grad
    for k=3,self.num_state+2 do table.insert(dstate[t-1], dinputs[k]) end

    if t ~= 1 then 
      local dcontext = dinputs[2]
      local datt = self.att_clones[t]:backward(self.att_inputs[t], dcontext)
      dstate[t-1][self.num_state]:add(datt[2])
      dinputs[1]:add(datt[3])
    end

    local dxt = dinputs[1]
    local it = self.lookup_tables_inputs[t]
    self.lookup_tables[t]:backward(it, dxt) 
  end

  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      local target_index
      if t > D then 
        target_index = 0
      else
        target_index = seq[{t,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
