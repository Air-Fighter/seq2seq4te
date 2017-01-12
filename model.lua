--
-- Creater: shawn
-- Date: 17-1-9
-- Time: 08:28
-- Function: Seq2Seq Model
--

torch.class("Seq2Seq")

function Seq2Seq:__init()
end

function Seq2Seq:build(opt, embeddings)
    local word_dim = embeddings:size(2)
    local emb_e = nn.LookupTableMaskZero(embeddings:size(1), word_dim)
    local emb_d = nn.LookupTableMaskZero(embeddings:size(1), word_dim)

    emb_e.weight:sub(2, -1):copy(embeddings)
    share_params(emb_e, emb_d)

    -- Encoder
    local enc = nn.Sequential()
    enc:add(emb_e)
    enc.lstmLayers = {}
    if opt.use_seqlstm then
        enc.lstmLayers[1] = nn.SeqLSTM(word_dim, opt.hidden_size)
        enc.lstmLayers[1]:maskZero()
        enc:add(enc.lstmLayers[1])
    else
        enc.lstmLayers[1] = nn.LSTM(word_dim, opt.hidden_size):maskZero(1)
        enc:add(nn.Sequencer(enc.lstmLayers[1]))
    end
    if opt.layer_nums > 1 then
        for i=2,opt.layer_nums do
            if opt.use_seqlstm then
                enc.lstmLayers[i] = nn.SeqLSTM(opt.hidden_size, opt.hidden_size)
                enc.lstmLayers[i]:maskZero()
                enc:add(enc.lstmLayers[i])
            else
                enc.lstmLayers[i] = nn.LSTM(opt.hidden_size, opt.hidden_size):maskZero(1)
                enc:add(nn.Sequencer(enc.lstmLayers[i]))
            end
        end
    end
    enc:add(nn.Select(1, -1))

    -- Decoder
    local dec = nn.Sequential()
    dec:add(emb_d)
    dec.lstmLayers = {}
    if opt.use_seqlstm then
        dec.lstmLayers[1] = nn.SeqLSTM(word_dim, opt.hidden_size)
        dec.lstmLayers[1]:maskZero()
        dec:add(dec.lstmLayers[1])
    else
        dec.lstmLayers[1] = nn.LSTM(word_dim, opt.hidden_size):maskZero(1)
        dec:add(nn.Sequencer(dec.lstmLayers[1]))
    end
    if opt.layer_nums > 1 then
        for i=2,opt.layer_nums do
            if opt.use_seqlstm then
                dec.lstmLayers[i] = nn.SeqLSTM(opt.hidden_size, opt.hidden_size)
                dec.lstmLayers[i]:maskZero()
                dec:add(dec.lstmLayers[i])
            else
                dec.lstmLayers[i] = nn.LSTM(opt.hidden_size, opt.hidden_size):maskZero(1)
                dec:add(nn.Sequencer(dec.lstmLayers[i]))
            end
        end
    end
    dec:add(nn.Sequencer(nn.MaskZero(nn.Linear(opt.hidden_size, opt.vocab_size), 1)))
    dec:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

    local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))

    self.enc = enc
    self.dec = dec
    self.criterion = criterion
    return enc, dec, criterion
end

function Seq2Seq:get_feval(encInSeq, decInSeq, decOutSeq, x, dl_dx, opt)
    local enc = self.enc
    local dec = self.dec
    local criterion = self.criterion

    self.feval = function(x_new)
        if x ~= x_new then x:copy(x_new) end
        dl_dx:zero()

        -- Forward pass
        local encOut = enc:forward(encInSeq)
        forwardConnect(enc, dec, opt)
        local decOut = dec:forward(decInSeq)
        local err = criterion:forward(decOut, decOutSeq)

        -- Backward pass
        local gradOutput = criterion:backward(decOut, decOutSeq)
        dec:backward(decInSeq, gradOutput)
        backwardConnect(enc, dec, opt)
        local zeroTensor = encOut:zero()
        enc:backward(encInSeq, zeroTensor)

        return err, dl_dx
    end

    return self.feval
end

function Seq2Seq:eval(encInSeq, decInSeq, decOutSeq, opt)
    -- This function is to return the loss of validation data
    local encOut = self.enc:forward(encInSeq)
    forwardConnect(self.enc, self.dec, opt)
    local decOut = self.dec:forward(decInSeq)
    local err = self.criterion:forward(decOut, decOutSeq)
    return err
end

function Seq2Seq:forward(encInSeq, opt)
    local enc = self.enc
    local dec = self.dec
    enc:forward(encInSeq)
    forwardConnect(enc, dec, opt)
    -- Initialize decoder inputs to 1 * samples_num filling with <SOS>
    local decoder_inputs = torch.Tensor(1, encInSeq:size(2)):fill(opt.sos)
    -- Initialize the outputs of decoder as samples_num * seq_len filling with 0
    local decoder_outputs = torch.Tensor(opt.seq_len, encInSeq:size(2)):zero()
    local sent_end = torch.IntTensor(encInSeq:size(2)):zero()

    for i=1, opt.seq_len do
        local decoder_output = dec:forward(decoder_inputs)
        -- Get most likely output
        local max_score, max_index = decoder_output[1]:max(2)
        max_index = max_index:t()

        for j=1, max_index:size(2) do
            if max_index[1][j] ~= opt.eos and sent_end[j] == 0 then
                decoder_outputs[i][j] = max_index[1][j]
            else
                sent_end[j] = 1
            end
        end

        -- All sentences are ended
        if sent_end:sum() == sent_end:size(1) then break end
        decoder_inputs = max_index
    end

    return decoder_outputs
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(enc, dec, opt)
    for i=1,#enc.lstmLayers do
        if opt.use_seqlstm then
            dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].output[opt.seq_len]
            dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cell[opt.seq_len]
        else
            dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[opt.seq_len])
            dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[opt.seq_len])
        end
    end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(enc, dec, opt)
    for i=1,#enc.lstmLayers do
        if opt.use_seqlstm then
            enc.lstmLayers[i].userNextGradCell = dec.lstmLayers[i].userGradPrevCell
            enc.lstmLayers[i].gradPrevOutput = dec.lstmLayers[i].userGradPrevOutput
        else
            enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
            enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
        end
    end
end
