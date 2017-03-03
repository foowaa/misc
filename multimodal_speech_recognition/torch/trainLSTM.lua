-- execute this file
--https://github.com/szagoruyko/cifar.torch/blob/master/train.lua
-- Load dependencies

-- 5 blocks: repr. block (DAE), imagining block(VAE), fusion block(mDAE/DBN), connection block(LSTM), classfier (SVM) 

local optim = require 'optim'
local gnuplot = require 'gnuplot'
--local nn = require 'nn'
local math = require 'math'
--local rnn = require 'rnn'
local cuda = pcall(require, 'cutorch')
local hasCudnn, cudnn = pcall(require, 'cudnn')

-- set up
print('Setting up LSTM')
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
--local label_test = data.label_test:transpose(1,2)
local video_train = data.video_train:transpose(1,2)
local audio_train = data.audio_train:transpose(1,2)
local label_train = data.label_train:transpose(1,2)
--local dd = hdf5.open('/home/user3/hhh.h5','r')
--local label_train = dd:read('label_train'):all()
--print(label_train:size())
--print(torch.type(label_train))
if cuda then
	video_train = video_train:cuda()
	audio_train = audio_train:cuda()
label_train = label_train:cuda()
	--video_test = video_test:cuda()
	audio_test = audio_test:cuda()
--label_test = label_test:cuda()
end

--local model11 = torch.load('./dae1.t7')
--local data1 = model11:get(1):forward(video_train)
--print(data1)
--local model12 = torch.load('./dae2.t7')
--local data2 = model12:get(1):forward(audio_train)
--local data_after_DAE = torch.cat(data1, data2)
local model2 = torch.load('./fusion1.t7')
local data_after_fusion = model2:get(1):forward(torch.cat(video_train,audio_train))
--print(torch.type(data_after_fusion))
--local model3 = torch.load('./vae.t7')
--local data3 = model3:
if cuda then 
  data_after_fusion:cuda()
end
--print(torch.type(data_after_fusion))
-- Due to VAE is to imagine, here doesn't need to forward through it

--print(data_after_fusion:size(2))


--print(video_train:size(1))
--Configuration
local cmd = torch.CmdLine()
--cmd:text()
cmd:option('-learningRate4LSTM','0.001','Learning Rate for VAE')
 --TODO: AEs, DBM, DBN ?
 --[[TODO: SVM config, I want to use python to do it, as the torch-svm only supports liblinear format files, but here is torch.tensor. Transfermation is uneconomical
 --]] 
cmd:option('-optimizer','adam','Optimizer')
cmd:option('-epochs4LSTM','20', 'epochs for LSTM')

--cmd:text()
local  opt = cmd:parse(arg)
opt.batchSize = 40  --Question: whether same batchSize is suitable?


--Create models
 --Create DAE
 local model = require('./LSTM')
 model:createLSTM(data_after_fusion)
 --print(model1)
 local model1 = model.lstm
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
 local criterion1 = nn.ClassNLLCriterion()
 --local criterion12 = nn.BCECriterion()
 if cuda then
  criterion1:cuda()
  --criterion12:cuda()
 end
print(parameters11:size())
--print(torch.type(criterion1))

--Train
 --train DAE1
 print('Train LSTM')
local optimParams = {learningRate = opt.learningRate4LSTM}
local epoch = 0
local loss
local losses = {}
local timer = torch.Timer()
local epochs = tonumber(opt.epochs4LSTM)
while epoch < epochs do
    epoch = epoch + 1
   -- local shuffle = torch.randperm(video_train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    --local indices = torch.randperm(data_after_fusion:size(1)):long():split(opt.batchSize)
	local si = data_after_fusion:size(1)
	local temp_tensor = torch.linspace(1,si,si)
	local indices = temp_tensor:long():split(opt.batchSize)
	
    indices[#indices] = nil
   -- local N = #indices * tonumber(opt.batchSize)
   --require 'cunn'
   local outputs = torch.FloatTensor(opt.batchSize):cuda()
   --print(torch.type(outputs))
--debug.debug()
   -- local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
		--print(t)
		--print(v)
        local inputs = data_after_fusion:index(1,v)  -- 会在opfunc中使用
        --print(torch.type(inputs))
        outputs:copy(label_train:index(1,v))
        --NOTE: label_train is 2-dim 
        --print(v)
        --print(inputs:size())
        --local outputs = label_train[{{v}}]
        --local tempSize = v:size()[1]
        --print(tempSize)
        --local outputs 
        --local tempTensor = label_train:index(1,v)
        --print(tempTensor:size())
        --print(torch.type(tempTensor))
        --local tempTable = {}
        --for i = 1,tempSize do
        --  tempTable[i] = tempTensor[i][1]
        --end
       -- outputs = torch.Tensor(tempTable)
       -- print(torch.type(outputs))
        --if cuda then
        --  outputs:cuda()
       -- end
       --print(torch.type(outputs))
        local opfunc = function(x)
            if x ~= parameters11 then
                parameters11:copy(x)
            end
            gradParameters11:zero()

            local xHat = model1:forward(inputs)
            local loss = criterion1:forward(xHat, outputs)
            local gradLoss = criterion1:backward(xHat, outputs)
            model1:backward(inputs, gradLoss)

            return loss, gradParameters11
        end

       -- print(opfunc)
       -- print(parameters11:size())
      -- print(optimParams)
        _, loss = optim[opt.optimizer](opfunc, parameters11, optimParams)  
        losses[#losses + 1] = loss

        --plot training curve
       local plots = {{'LSTM', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}} 
       gnuplot.plot(table.unpack(plots))
       gnuplot.ylabel('Loss')
       gnuplot.xlabel('Batch')
    end
end

print('training time of Fuison:' .. timer:time().real ..'s')



torch.save('./lstm2.t7',model1)



