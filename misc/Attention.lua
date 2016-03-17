require 'nn'
require 'nngraph'
local utils = require 'misc.utils'

local Attention = {}
function Attention.att(batch_size, seq_per_img, num_annotations, annotation_size, embedding_size, rnn_size, dropout)

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- imgs
  table.insert(inputs, nn.Identity()()) -- h

  local i2l1 = nn.Linear(annotation_size+rnn_size,embedding_size)
  local i2l2 = nn.Linear(embedding_size,1)
  -- local a2h = nn.Linear(annotation_size, embedding)
  -- local h2h = nn.Linear(rnn_size, embedding)
  -- local proj = nn.Linear(embedding, 1)

  local cvectors = {}
  for b=1,batch_size do
    local img = nn.Select(1,b)(inputs[1]) -- num_annotations x annotation_size
    local h = nn.Select(1,(b-1)*seq_per_img+1)(inputs[2]) -- rnn_size
    -- local repl = nn.Replicate(num_annotations, 1)(h)

    local i = nn.JoinTable(2)({img,nn.Replicate(num_annotations,1)(h)}) -- num_annotations x (annotation_size + rnn_size)
    local l1 = nn.Tanh()(nn.Linear(annotation_size+rnn_size,embedding_size):share(i2l1,'weight', 'bias', 'gradWeight', 'gradBias')(i))
    if dropout > 0 then l1 = nn.Dropout(dropout)(l1) end
    local l2 = nn.Linear(embedding_size,1):share(i2l2,'weight', 'bias', 'gradWeight', 'gradBias')(l1) -- num_annotations x 1
    local probs = nn.SoftMax()(nn.View(num_annotations)(l2))
    
    -- local sum = nn.Tanh()(nn.CAddTable()({
    --     nn.Linear(annotation_size, embedding):share(a2h, 'weight', 'bias', 'gradWeight', 'gradBias')(img),
    --     nn.Linear(rnn_size, embedding):share(h2h, 'weight', 'bias', 'gradWeight', 'gradBias')(repl)
    --   }))
    -- local proj = nn.Linear(embedding, 1):share(proj, 'weight', 'bias', 'gradWeight', 'gradBias')(sum)
    -- local probs = nn.SoftMax()(nn.View(num_annotations)(proj))    

    local weighted_annotations = nn.CMulTable()({img, nn.Replicate(annotation_size,2)(probs)})
    local context_vector = nn.Replicate(seq_per_img)(
                            (nn.Sum(1)(weighted_annotations)))
    table.insert(cvectors, context_vector)
  end

  local context = nn.JoinTable(1)(cvectors)
  table.insert(outputs, context)

  return nn.gModule(inputs, outputs)
end

return Attention