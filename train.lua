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
cmd:text("--------Running Options--------")
cmd:option('-gpu_id', 3, [[Id of which gpu is used to train the model, -1 is to use CPU]])
cmd:option('-valid_freq', 5, [[Frequency of validation]])
cmd:option('-save_freq', 5, [[Frequency of saving parameters]])
cmd:text("--------Data Options--------")
cmd:option('-data_prefix','./data/small-', [[Path to the training small-train.txt and small-dev.txt]])
cmd:option('-embedding_file', './data/small-wordembedding', [[Path to the pre-trained embedding file]])
cmd:option('-save_file', './data/params/model_', [[Prefix of the file that tends to save the parameters]])
cmd:text("--------Optimizer Options--------")
cmd:option('-learning_rate', 1e-2, [[The learning rate for optimizer]])
cmd:option('-learning_rate_decay', 1e-4, [[Decay of learning rates]])
cmd:option('-weight_decay', 1e-3, [[The decay of parameters]])
cmd:option('-momentum', 1e-4, [[Momentum for optimizer]])
cmd:text("--------Model Options--------")
cmd:option('-relation', 'entail', [[Which relationship is the aim of training]])
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

print(string.format('%d words loaded into system.', #dict.vocab))
opt.vocab_size = #dict.vocab
opt.seq_len = snli.max_len
opt.sos = dict:get_word_idx("<SOS>")
opt.eos = dict:get_word_idx("<EOS>")

optim_configs = {
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum
}

local seq2seq_entail = Seq2Seq()
local enc, dec, criterion = seq2seq_entail:build(opt, dict.embeddings)
if opt.gpu_id > 0 then
    local cutorch = require 'cutorch'
    local cunn = require 'cunn'
    cutorch.setDevice(opt.gpu_id)
    enc = enc:cuda()
    dec = dec:cuda()
    criterion = criterion:cuda()
end

-- Concatenate the enc's and dec's parameters
local x, dl_dx = nn.Container():add(enc):add(dec):getParameters()


for iter=1,opt.iter_nums do
    local err = 0
    print(string.format("Iteration %d: ", iter))
    local start_time = os.clock()
    io.write("\tProgress: ")
    for i=1, snli:get_set_size('train', opt.relation, dict), opt.batch_size do
        draw_progress_bar(i, snli:get_set_size('train', opt.relation, dict), 40)
        -- Get data from SNLI:get_batch_data, the parameters in order are:
        -- 'train'/'dev', 'entail'/'neutral'/'contradict', dict, index of batch start, batch size
        local encInSeq, decInSeq, decOutSeq = nil, nil, nil
        if opt.gpu_id > 0 then
            encInSeq, decInSeq, decOutSeq =
                snli:get_batch_data('train', opt.relation, dict, i, opt.batch_size, true)
        else
            encInSeq, decInSeq, decOutSeq =
                snli:get_batch_data('train', opt.relation, dict, i, opt.batch_size, false)
        end
        local feval = seq2seq_entail:get_feval(encInSeq, decInSeq, decOutSeq, x, dl_dx, opt)
        local _, fs = optim.adadelta(feval, x, optim_configs)
        err = err + fs[1]
    end
    io.write('\n')
    print(string.format("\tNLL err = %f ", iter, err))
    local end_time = os.clock()
    print(string.format("\tRun time: %f seconds", end_time - start_time))

    err = 0
    if iter % opt.valid_freq == 0 then
        snli:reset_batch_data()
        for i=1, snli:get_set_size('dev', opt.relation, dict), opt.batch_size do
            -- Get data from SNLI:get_batch_data, the parameters in order are:
            -- 'train'/'dev', 'entail'/'neutral'/'contradict', dict, index of batch start, batch size
            local encInSeq, decInSeq, decOutSeq = nil, nil, nil
            if opt.gpu_id > 0 then
                encInSeq, decInSeq, decOutSeq =
                    snli:get_batch_data('dev', opt.relation, dict, i, opt.batch_size, true)
            else
                encInSeq, decInSeq, decOutSeq =
                    snli:get_batch_data('dev', opt.relation, dict, i, opt.batch_size, false)
            end
            err = err + seq2seq_entail:eval(encInSeq, decInSeq, decOutSeq, opt)
        end
        print(string.format("\tValidating result: NLL err = %f ", iter, err))
        snli:reset_batch_data()
    end

    if iter % opt.save_freq == 0 then
        print(string.format('\tSaving parameters to %sepoch%.2f_%.2f.t7', opt.save_file, iter, err))
        local savefile = string.format('%sepoch%.2f_%.2f.t7', opt.save_file, iter, err)
        torch.save(savefile, {{seq2seq_entail.enc, seq2seq_entail.dec}, opt})
    end
end

print(string.format('\tSaving parameters to %sfinal.t7', opt.save_file, iter, err))
local savefile = string.format('%sfinal.t7', opt.save_file)
torch.save(savefile, {{seq2seq_entail.enc:double(), seq2seq_entail.dec:double()}, opt})