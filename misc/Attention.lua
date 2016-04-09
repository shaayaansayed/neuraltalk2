require 'nn'
require 'nngraph'
local utils = require 'misc.utils'

local Attention = {}
function Attention.att(batch_size, num_annotations, annotation_size, rnn_size, input_size, dropout)

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- imgs: batch_size x num_annotations x annotation_size
  table.insert(inputs, nn.Identity()()) -- h_prev: batch_size x rnn_size
  table.insert(inputs, nn.Identity()()) -- xt: batch_size x input_size

  local img = inputs[1]
  local h_prev = inputs[2]
  local xt = inputs[3]

  local img_flatten = nn.View(batch_size*num_annotations,annotation_size)(img)
  local att_img = nn.Linear(annotation_size,rnn_size)(img_flatten)
  local att_h = nn.Linear(rnn_size,rnn_size)(h_prev)
  local att_xt = nn.Linear(input_size,rnn_size)(xt)

  -- batch_size x num_annotations x rnn_size
  local att_img_expand = nn.View(batch_size, num_annotations, rnn_size)(att_img) 
  local att_h_expand = nn.View(batch_size, 1, rnn_size)(att_h)
  att_h_expand = nn.Replicate(num_annotations, 2)(att_h_expand) -- batch_size x num_annotations x annotation_size
  local att_xt_expand = nn.View(batch_size, 1, rnn_size)(att_xt)
  att_xt_expand = nn.Replicate(num_annotations, 2)(att_xt_expand)

  local sum = nn.Tanh()(nn.CAddTable()({att_img_expand, att_h_expand, att_xt_expand}))
  sum = nn.View(batch_size*num_annotations, rnn_size)(sum)
  local annotation_scores = nn.Linear(rnn_size, 1)(sum) -- batch_size*num_annotations x 1 
  annotation_scores = nn.View(batch_size, -1)(annotation_scores) -- batch_size x num_annotations
  annotation_scores = nn.SoftMax(2)(annotation_scores):annotate{name='scores'} -- batch_size x num_annotations
  annotation_scores = nn.Replicate(annotation_size, 3)(annotation_scores) -- batch_size x num_annotations x annotation_size

  local weighted_annotations = nn.CMulTable()({annotation_scores, img})
  local context_vectors = nn.Sum(2)(weighted_annotations)

  table.insert(outputs, context_vectors)

  return nn.gModule(inputs, outputs)
end

return Attention