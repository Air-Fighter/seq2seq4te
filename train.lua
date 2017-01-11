--
-- Creater: shawn
-- Date: 17-1-9
-- Time: 上午8:20
-- Function: 
--

--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'torch'
require 'rnn'
require 'optim'
require 'utils'
require 'model'
require 'wordembedding'
require 'snli'

cmd = torch.CmdLine()

cmd:text("--------Data Options    --------")
cmd:option('-data_prefix','./data/small-', [[Path to the training small-train.txt and small-dev.txt]])
cmd:option('-embedding_file', './data/small-wordembedding', [[Path to the pre-trained embedding file]])
cmd:text("--------Optimizer Options--------")
cmd:option('-learning_rate', 1e-2, [[The learning rate for optimizer]])
cmd:option('-learning_rate_decay', 1e-4, [[Decay of learning rates]])
cmd:option('-weight_decay', 1e-3, [[The decay of parameters]])
cmd:option('-momentum', 1e-4, [[Momentum for optimizer]])
cmd:text("--------Model Options--------")
cmd:option('-hidden_size', 400, [[The dimensions of hidden units]])
cmd:option('-layer_nums', 1, [[The number of model's layers]])
cmd:option('-use_seqlstm', true, [[Use SeqLSTM instead of LSTM or not]])
cmd:option('-iter_nums', 10, [[Number of iterations]])
cmd:option('-batch_size', 2, [[Size of every batch]])

local opt = cmd:parse(arg)

local dict = WordEmbedding(opt.embedding_file)
local snli = SNLI(opt.data_prefix, 0, true, true)

dict:trim_by_counts(snli.word_counts)
dict:extend_by_counts(snli.train_word_counts)

opt.vocab_size = #dict.vocab
opt.seq_len = 7 -- length of the encoded sequence (with padding)
opt.sos = dict:get_word_idx("<SOS>")
opt.eos = dict:get_word_idx("<EOS>")

optim_configs = {
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum
}


local enc, dec = Seq2Seq:build(opt, dict.embeddings)
-- Concatenate the enc's and dec's parameters
local x, dl_dx = nn.Container():add(enc):add(dec):getParameters()


for iter=1,opt.iter_nums do
    local err = 0

    for i=1, 2, opt.batch_size do
        -- Get data from SNLI:get_batch_data, the parameters in order are:
        -- 'train'/'dev', 'entail'/'neutral'/'contradict', dict, index of batch start, batch size
        local encInSeq, decInSeq, decOutSeq = snli:get_batch_data('dev', 'entail', dict, i, opt.batch_size)
        local feval = Seq2Seq:get_feval(encInSeq, decInSeq, decOutSeq, x, dl_dx, opt)
        local _, fs = optim.adadelta(feval, x, optim_configs)
        err = err + fs[1]
    end

    print(string.format("Iteration %d ; NLL err = %f ", iter, err))
end

local encInSeq = torch.Tensor({{0,0,0,0,4,5,6}}):t()
local test_out = Seq2Seq:forward(encInSeq, opt)
print(test_out)