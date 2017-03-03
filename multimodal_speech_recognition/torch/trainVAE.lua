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
-- local data1 = model11:get(1):forward(video_train):round()
-- print(data1:size())
-- local model12 = torch.load('./dae2.t7')
-- local data2 = model12:get(1):forward(audio_train):round()
-- print(data2:size())
local data_after_DAE = torch.cat(video_train, audio_train)
-- print(data_after_DAE:size())
-- print(torch.type(data_after_DAE))
if cuda then
   data_after_DAE = data_after_DAE:cuda()
 end
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
cmd:option('-epochs4VAE','10', 'epochs for DAE')

--cmd:text()
local  opt = cmd:parse(arg)
opt.batchSize = 100  --Question: whether same batchSize is suitable?


--Create models
 --Create DAE
 local model = require('./VAE')
 model:createVAE(data_after_DAE)
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
 local criterion1 = nn.MSECriterion()
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
---------------------------------------------------------------
            --Regularization phase
            local encoder = model.encoder 
            --print(encoder)
            local nElements = xHat:nElement()
            --print(nElements)
            local mean, logVar = table.unpack(encoder.output)
           -- print(mean)
           -- print(logVar)
            local var = torch.exp(logVar)
            local KLLoss = -0.5 * torch.sum(1 + logVar - torch.pow(mean, 2) - var)
           -- print(KLLoss)
            KLLoss = KLLoss / nElements
            loss = loss + KLLoss
			local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}
			--print(gradKLLoss)
			--print(x:size())
			encoder:backward(inputs, gradKLLoss)
-----------------------------------------------------------------
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

print('training time of VAE:' .. timer:time().real ..'s')

torch.save('./vae1.t7',model1)

local dd = torch.cat(video_test[{{1},{}}],torch.zeros(1,200):cuda())
local d = model1:forward(dd)
print(d[{{1},{101,300}}])