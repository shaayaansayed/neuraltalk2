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
  local dropout = utils.getopt(opt, 'dropout', 0)

  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.seq_per_img = utils.getopt(opt, 'seq_per_img')
  self.batch_size = utils.getopt(opt, 'batch_size')
  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, 512, self.rnn_size, self.batch_size*self.seq_per_img, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  -- self.att = Attention.att(self.batch_size, self.seq_per_img, 196, 512, 512, self.rnn_size, dropout)
  self.att = Attention.att(self.batch_size*self.seq_per_img, 196, 512, self.rnn_size, self.input_encoding_size, dropout)
  -- self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  self.c_tape = {}
  self.h_tape = {}

  self.c_tape[1] = torch.zeros(batch_size, self.rnn_size)
  self.h_tape[1] = torch.zeros(batch_size, self.rnn_size)
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
      context = self.att_clones[t]:forward{feats, self.h_tape[#self.h_tape], xt}
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    local inputs = {xt,context,utils.clone_list(self.c_tape),utils.clone_list(self.h_tape)}
    local out = self.core:forward(inputs)
    table.insert(self.c_tape, out[1])
    table.insert(self.h_tape, out[2])
    logprobs = out[#out] 
  end

  return seq, seqLogprobs
end

function layer:updateOutput(input)
  local seq = input[3]
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0

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
      xt = self.lookup_tables[t]:forward(it) 
    else
      local it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        can_skip = true 
      end
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)

        local feats = input[2]
        self.att_inputs[t] = {feats, self.h_tape[#self.h_tape], xt}
        context = self.att_clones[t]:forward(self.att_inputs[t])
      end
    end

    if not can_skip then
      self.inputs[t] = {xt,context,utils.clone_list(self.c_tape),utils.clone_list(self.h_tape)}
      local out = self.clones[t]:forward(self.inputs[t])
      table.insert(self.c_tape, out[1])
      table.insert(self.h_tape, out[2])
      self.output[t] = out[3] 
      self.tmax = t
    end
  end

  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dimgs -- grad on input images

  -- go backwards and lets compute gradients
  local d_ctape = utils.clone_list(self.c_tape, true)
  local d_htape = utils.clone_list(self.h_tape, true)

  for t=self.tmax,1,-1 do
    local dout = {}
    table.insert(dout, d_ctape[t+1])
    table.insert(dout, d_htape[t+1])
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)

    for k=1,t do
      d_ctape[k]:add(dinputs[3][k])
      d_htape[k]:add(dinputs[4][k])
    end

    if t ~= 1 then 
      local dcontext = dinputs[2]
      local datt = self.att_clones[t]:backward(self.att_inputs[t], dcontext)
      d_htape[t]:add(datt[2])
      dinputs[1]:add(datt[3])
    end

    local dxt = dinputs[1]
    local it = self.lookup_tables_inputs[t]
    self.lookup_tables[t]:backward(it, dxt) 
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
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

      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
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
