require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
local net_utils = require 'misc.net_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert Images to Convnet Features')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','/scratch/cluster/vsub/ssayed/MSCOCO/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/scratch/cluster/vsub/ssayed/MSCOCO/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','/scratch/cluster/vsub/ssayed/cnn_proto/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','/scratch/cluster/vsub/ssayed/cnn_proto/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-feats_dir','/scratch/cluster/vsub/ssayed/MSCOCO/feats_nt_avg','where to save feat t7 files')

-- misc
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

local protos = {}

-- initialize the ConvNet
local cnn_backend = opt.backend
if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

local splits = {'train', 'val'}
for k,split_ix in pairs(splits) do
	local stop = false

	local batch_ix = 0
	local total_batches = math.ceil(#loader.split_ix[split_ix]/opt.batch_size)
	local folder = path.join(opt.feats_dir, split_ix)

	repeat 
		batch_ix = batch_ix + 1
		print(split_ix .. ' ' .. batch_ix .. '/' .. total_batches)

		local data = loader:getBatch{batch_size = opt.batch_size, split = split_ix, seq_per_img = 1}
		data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) 
		local feats = protos.cnn:forward(data.images)
		feats = feats:float()

		local filename = path.join(folder, batch_ix .. '.t7')
		torch.save(filename, feats)

		stop = data.bounds['wrapped']
	until stop

	assert(batch_ix == total_batches, 'something is wrong')
end
