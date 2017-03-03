-- execute this file

-- Load dependencies

-- 5 blocks: repr. block (DAE), imagining block(VAE), fusion block(mDAE/DBN), connection block(LSTM), classfier (SVM) 

local optim = require 'optim'
local gnuplot = require 'gnuplot'
--local nn = require 'nn'
local math = require 'math'
--local rnn = require 'rnn'
local cuda = pcall(require, 'cutorch')
local hasCudnn, cudnn = pcall(require, 'cudnn')
require 'nn'
require 'dpnn'
-- set up
print('Setting up VAE')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
    require 'cunn'
    cutorch.manualSeed(torch.random())
end

--load data from read_file
print('Loading data')
require 'lfs'
lfs.chdir('/home/user3/work')
 --debug.debug()
local data = require './read_file'
data:load_data('/home/user3/data_train.h5','/home/user3/data_test.h5')
local video_test = data.video_test:transpose(1,2)
local audio_test = data.audio_test:transpose(1,2)
local label_test = data.label_test:transpose(1,2)
local video_train = data.video_train:transpose(1,2)
local audio_train = data.audio_train:transpose(1,2)
local label_train = data.label_train:transpose(1,2)
if cuda then
	video_train = video_train:cuda()
	audio_train = audio_train:cuda()
	label_train = label_train:cuda()
	video_test = video_test:cuda()
	audio_test = audio_test:cuda()
	label_test = label_test:cuda()
end

-- local model11 = torch.load('./dae1.t7')
-- local data1 = model11:get(1):forward(video_train)
--print(data1)
-- local model12 = torch.load('./dae2.t7')
-- local data2 = model12:get(1):forward(audio_train)
local data_after_DAE = torch.cat(video_train, audio_train)
local model3 = torch.load('./vae1.t7')
local data3 = model3

--Due to VAE is to imagine, here doesn't need to forward through it
-- if cuda then
  -- data_after_DAE = data_after_DAE:cuda()
-- end
--print(data_after_DAE:size(2))


--print(video_train:size(1))
--Configuration
local cmd = torch.CmdLine()
--cmd:text()
cmd:option('-learningRate4VAE','0.001','Learning Rate for VAE')
 --TODO: AEs, DBM, DBN ?
 --[[TODO: SVM config, I want to use python to do it, as the torch-svm only supports liblinear format files, but here is torch.tensor. Transfermation is uneconomical
 --]] 
cmd:option('-optimizer','adam','Optimizer')
cmd:option('-epochs4VAE','20', 'epochs for DAE')

--cmd:text()
local  opt = cmd:parse(arg)
opt.batchSize = 100  --Question: whether same batchSize is suitable?


--Create models
 --Create DAE
 local model = require('./Fusion')
 model:createFusion(data_after_DAE)
 --print(model1)
 local model1 = model.autoencoder
 --print(model1)
 --model1 = require('./DAE')
 --print(model1)
 if cuda then
  model1:cuda()
  if hasCudnn then
    cudnn.convert(model1,cudnn)
  end
 end
  --print(model1)
  

 --Create VAE


 --Create LSTM, see above
 
 

-- Get paramethers, create loss
 --DAE
 local parameters11, gradParameters11 = model1:getParameters()
 local criterion1 = nn.BCECriterion()
 --local criterion12 = nn.BCECriterion()
 if cuda then
  criterion1:cuda()
  --criterion12:cuda()
 end
--print(parameters11)

 --SVM, see above


--Train
 --train DAE1
 print('Train VAE')
local optimParams = {learningRate = opt.learningRate4VAE}
local epoch = 0
local loss
local losses = {}
local timer = torch.Timer()
local epochs = tonumber(opt.epochs4VAE)
while epoch<epochs do
    epoch = epoch + 1
    --local lowerbound = 0
    --local tic = torch.tic()
    --local losses = {}
   -- local shuffle = torch.randperm(video_train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(data_after_DAE:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil
   -- local N = #indices * tonumber(opt.batchSize)
   
--debug.debug()
   -- local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = data_after_DAE:index(1,v)  -- 会在opfunc中使用

        local opfunc = function(x)
            if x ~= parameters11 then
                parameters11:copy(x)
            end
            gradParameters11:zero()

            local xHat = model1:forward(inputs)
            local loss = criterion1:forward(xHat, inputs)
            local gradLoss = criterion1:backward(xHat, inputs)
            model1:backward(inputs, gradLoss)

            return loss, gradParameters11
        end

       -- print(opfunc)
        --print(parameters11:size())
       -- print(optimParams)
        _, loss = optim[opt.optimizer](opfunc, parameters11, optimParams)  
        losses[#losses + 1] = loss

        --plot training curve
       local plots = {{'VAE', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}} 
       gnuplot.plot(table.unpack(plots))
       gnuplot.ylabel('Loss')
       gnuplot.xlabel('Batch')
    end

end

print('training time of Fuison:' .. timer:time().real ..'s')

torch.save('./fusion1.t7',model1)




-- y = data_after_DAE:double()
-- nn.utils.recursiveType(model1, 'torch.CudaTensor')
-- print(torch.type(y))
-- print(torch.type(model1))




local x = model1:get(1):forward(data_after_DAE)
x=x:double()
--print(x)
require 'hdf5'
local myFile = hdf5.open('./result_torch.h5','w')
myFile:write('data',x)
myFile:close()
