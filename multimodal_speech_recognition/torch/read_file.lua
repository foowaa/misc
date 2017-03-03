--load data
require 'hdf5'
local cuda = pcall(require, 'cutorch')

local data={}
function data:load_data(path_train, path_test)
	--local data = {}
	local trainFile = hdf5.open(path_train,'r')
	local testFile = hdf5.open(path_test,'r')
	self.video_train = trainFile:read('video_train'):all()
	self.audio_train = trainFile:read('audio_train'):all()
	self.label_train = trainFile:read('label_train'):all()
	self.video_test = testFile:read('video_test'):all()
	self.audio_test = testFile:read('audio_test'):all()
	self.label_test = testFile:read('label_test'):all()
    trainFile:close()
    testFile:close()
	--if cuda then
    --	self.video_train = self.video_train:cuda()
    	--self.audio_train = self.audio_train:cuda()
    	--self.label_train = self.label_train:cuda()

    	--self.video_test = self.video_test:cuda()
    	--self.audio_test = self.audio_test:cuda()
    	--self.label_test = self.label_test:cuda()
	--end
	--return data
end
return data
	
--[[
require 'lfs'
lfs.chdir('/home/user3/test')
--debug.debug()
local data=require './read_file'
data:load_data('/home/user3/data_train.h5','/home/user3/data_test.h5')
local video_test=data.video_test
print(video_test)

]]