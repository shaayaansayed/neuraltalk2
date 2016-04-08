require 'nn'
require 'nngraph'
require 'misc.ReplicateAdd'

local LSTM = {}
function LSTM.lstm(input_size, output_size, context_size, rnn_size, batch_size, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  table.insert(inputs, nn.Identity()()) -- context
  table.insert(inputs, nn.Identity()()) -- memory tape
  table.insert(inputs, nn.Identity()()) -- hidden tape 

  local h_tape = nn.JoinTable(2)(inputs[4]) -- batch_size x rnn_size*timesteps
  local att_x = nn.Linear(input_size,rnn_size)(inputs[1]) -- batch_size x rnn_size
  local att_prevh = nn.Linear(rnn_size, rnn_size)(nn.View(-1, rnn_size)(h_tape)) -- view: batch_size*timesteps x rnn_size -- linear: batch_size*timesteps x rnn_size
  att_prevh = nn.View(batch_size, -1)(att_prevh)
  local mem_sum = nn.Tanh()(nn.ReplicateAdd()({att_prevh, att_x}))
  local mem_scores = nn.Linear(rnn_size, 1)(nn.View(-1, rnn_size)(mem_sum))
  mem_scores = nn.SoftMax(2)(mem_scores) 
  mem_scores = nn.View(batch_size, 1, -1)(mem_scores)

  h_tape = nn.View(batch_size, -1, rnn_size)(h_tape)
  local c_tape = nn.JoinTable(2)(inputs[3]) -- batch_size x rnn_size*timesteps
  c_tape = nn.View(batch_size, -1, rnn_size)(c_tape)
  local att_h = nn.View(batch_size, rnn_size)(nn.MM(false, false)({mem_scores, h_tape}))  --this is the allignment vector at time step t
  local att_c = nn.View(batch_size, rnn_size)(nn.MM(false, false)({mem_scores, c_tape}))

  local x, input_size_L
  local outputs = {}
  x = inputs[1]
  input_size_L = input_size

  local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'}
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(att_h):annotate{name='h2h_'}
  local c2h = nn.Linear(context_size, 4 * rnn_size)(inputs[2]):annotate{name='h2h_'}
  local all_input_sums = nn.CAddTable()({i2h, h2h, c2h})

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)

  local in_transform = nn.Tanh()(n4)
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, att_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

