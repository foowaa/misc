-- execute this file

-- Load dependencies

-- 5 blocks: repr. block (DAE), imagining block(VAE), fusion block(mDAE/DBN), connection block(LSTM), classfier (SVM) 

local optim = require 'optim'
local gnuplot = require 'gnuplot'
--local nn = require 'nn'
local math = require 'math'
--local rnn = require 'rnn'
local h5 = require 'hdf5'
local cuda = pcall(require, 'cutorch')
local hasCudnn, cudnn = pcall(require, 'cudnn')

-- set up
print('Setting up DAE')
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
--print(video_train:size(1))
--Configuration
local cmd = torch.CmdLine()
--cmd:text()
cmd:option('-learningRate4DAE','0.001','Learning Rate for DAE')
 --TODO: AEs, DBM, DBN ?
 --[[TODO: SVM config, I want to use python to do it, as the torch-svm only supports liblinear format files, but here is torch.tensor. Transfermation is uneconomical
 --]] 
cmd:option('-optimizer','adam','Optimizer')
cmd:option('-epochs4DAE','10', 'epochs for DAE')

--cmd:text()
local  opt = cmd:parse(arg)
opt.batchSize = 100  --Question: whether same batchSize is suitable?


--Create models
 --Create DAE
 local model1 = require('./DAE1')
 model1:createDAE(video_train)
 --print(model1)
 local model11 = model1.autoencoder
 --print(model11)
 --model1 = require('./DAE')
 local model2 = require('./DAE2')
 model2:createDAE(audio_train)
 local model12 = model2.autoencoder
 --print(model11)
 if cuda then
  model11:cuda()
  model12:cuda()
  if hasCudnn then
    cudnn.convert(model11,cudnn)
    cudnn.convert(model12,cudnn)
  end
 end
  --print(model12)
  

 --Create VAE


 --Create LSTM, see above
 
 

-- Get paramethers, create loss
 --DAE
 local parameters11, gradParameters11 = model11:getParameters()
 local parameters12, gradParameters12 = model12:getParameters()
 local criterion1 = nn.BCECriterion()
 --local criterion12 = nn.BCECriterion()
 if cuda then
  criterion1:cuda()
  --criterion12:cuda()
 end
--print(parameters11)
print(parameters12:size())
 --SVM, see above


--Train
 --train DAE1
 print('Train DAE')
local optimParams = {learningRate = opt.learningRate4DAE}
local epoch = 0
local loss
local losses = {}
local timer1 = torch.Timer()
local epochs = tonumber(opt.epochs4DAE)
while epoch<epochs do
    epoch = epoch + 1
    --local lowerbound = 0
    --local tic = torch.tic()
    --local losses = {}
   -- local shuffle = torch.randperm(video_train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(video_train:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil
   -- local N = #indices * tonumber(opt.batchSize)
   
--debug.debug()
   -- local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = video_train:index(1,v)  -- 会在opfunc中使用

        local opfunc = function(x)
            if x ~= parameters11 then
                parameters11:copy(x)
            end
            gradParameters11:zero()

            local xHat = model11:forward(inputs)
            local loss = criterion1:forward(xHat, inputs)
            local gradLoss = criterion1:backward(xHat, inputs)
            model11:backward(inputs, gradLoss)

            return loss, gradParameters11
        end
       -- print(opfunc)
        --print(parameters11:size())
       -- print(optimParams)
        _, loss = optim[opt.optimizer](opfunc, parameters11, optimParams)  
        losses[#losses + 1] = loss

        --plot training curve
       --local plots = {{'DAE2', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}} 
       --gnuplot.plot(table.unpack(plots))
       --gnuplot.ylabel('Loss')
       --gnuplot.xlabel('Batch')
    end

end

print('training time of DAE1:' .. timer1:time().real ..'s')
--print("Training DAE1 Time: ".. timer1:time())
torch.save('./dae1.t7',model11)
--print(model11)
--local xHat = model11:forward(video_train) 
--print(model11:get(1):forward(video_train):size())

--debug.debug()
-- model12
epoch = 0
loss = 0
losses = {}
optimParams = {learningRate = opt.learningRate4DAE}
local timer2 = torch.Timer()
--local epochs = tonumber(opt.epochs4DAE)
while epoch<epochs do
    epoch = epoch + 1
    --local lowerbound = 0
    --local tic = torch.tic()
    --local losses = {}
    --local shuffle = torch.randperm(audio_train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(audio_train:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil
  --  local N = #indices * tonumber(opt.batchSize)
 --debug.debug()   
--debug.debug()

    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = audio_train:index(1,v)  -- 会在opfunc中使用

        local opfunc = function(x)
            if x ~= parameters12 then
                parameters12:copy(x)
            end
            gradParameters12:zero()

            local xHat = model12:forward(inputs)
            local loss = criterion1:forward(xHat, inputs)
            local gradLoss = criterion1:backward(xHat, inputs)
            model12:backward(inputs, gradLoss)

            return loss, gradParameters12
        end
    --debug.debug()
        --print(opfunc)
        --print(parameters12:size())
        --print(optimParams)
        _, loss = optim[opt.optimizer](opfunc, parameters12, optimParams)  
        losses[#losses + 1] = loss

        --plot training curve
       local plots = {{'DAE2', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}} 
       gnuplot.plot(table.unpack(plots))
       gnuplot.ylabel('Loss')
       gnuplot.xlabel('Batch')
    end
-- TODO : test
end

print('training time of DAE2:' .. timer2:time().real ..'s')
--print("Training DAE1 Time: ".. timer1:time())
torch.save('./dae2.t7',model12)
 --train VAE
 
 --load data from DAE
--local model11f = mdoel1:createDAE(video_train)


 --train fusion
 --local xx = torch.load('dae1.t7')
 --print(xx)
 -- LSTM


 --forward data, saving at the same time 

--local x = model11:get(1):forward(video_test):round()
--print(x)

 