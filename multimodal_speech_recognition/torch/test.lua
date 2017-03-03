--testing my model
--local optim = require 'optim'
local gnuplot = require 'gnuplot'
local nn = require 'nn'
local math = require 'math'
require 'nn'
require 'dpnn'
require 'rnn'
--local rnn = require 'rnn'
local cuda = pcall(require, 'cutorch')
local hasCudnn, cudnn = pcall(require, 'cudnn')

-- set up
print('Starting testing!')
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

print('Forwarding(A&V)...')
--forward all data here
--local model1 = torch.load('./dae1.t7')
--local data1 = model1:get(1):forward(video_test)
--local model2 = torch.load('./dae2.t7')
--local data2 = model2:get(1):forward(audio_test)
--local data2 = model2:get(1):forward(torch.zeros(2400,200):cuda())
local model3 = torch.load('./vae1.t7')
local data3 = model3:forward(torch.cat(video_test,torch.zeros(2400,200):cuda()))
local model4 = torch.load('./fusion1.t7')
local data41 = model4:get(1):forward(torch.cat(video_test,audio_test))
local data42 = model4:get(1):forward(data3)
local model5 = torch.load('./lstm1.t7')
local data51 = model5:forward(data41)
local data52 = model5:forward(data42)
--print(data51:size())
--local data51 = torch.load('./data.t7')

print('Evaluating(A&V)...')
data51 = torch.exp(data51)
data52 = torch.exp(data52)
local num_test = 60
local datat1 = {}
local datat2 = {}
local datat3 = {}
local accuracy = torch.zeros(num_test)
local result1 = {}
local result2 = {}
local result3 = {}
local step = 40
--evaluation
torch.save('./data.t7',data51)
-- A&V
for i = 1,60 do
	datat1[i] = torch.zeros(10):cuda()
  datat2[i] = torch.zeros(10):cuda()
  datat3[i] = torch.zeros(10):cuda()
  --print(datat)
	for j = 1,step do
		--data[i] = data51[{{j+(i-1)*step},{}}] + data[i]    --mean pooling
    --print(j+(i-1)*step)
    --print(data51)
    local x = j+(i-1)*step
    
		datat1[i] = data51[x] + datat1[i]
    datat2[i] = data52[x] + datat2[i]
	end
	datat1[i] = datat1[i]/step
  datat2[i] = datat2[i]/step
	_,result1[i] = torch.max(datat1[i],1)
  _,result2[i] = torch.max(datat2[i],1)
end


--torch.save('./result1.txt',result1,'ascii')
--torch.save('./result2.txt',result2,'ascii')
local res
for i=40,2400,60 do
  --res = result1[i]
  _,res = torch.max(data51[i],1)
  print(i)
  print(res)
end

print('------------')
for i=40,2400,60 do
  --res = data51[i]
  _,res = torch.max(data52[i],1)
  print(i)
 print(res)
end

