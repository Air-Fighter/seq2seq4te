--[[

    Loads SNLI entailment dataset.

--]]

require "torch"
require "pl"
local moses = require("moses")
require 'math'

require"utils"

torch.class("SNLI")

function SNLI:__init(snli_path_prefix, train_size, lower_case, verbose)
    self.num_relations = 3
    self.relations = {["contradiction"] = 1, ["neutral"] = 2, ["entailment"] = 3}
    self.rev_relations = {}
    for r, i in pairs(self.relations) do self.rev_relations[i] = r end
    self.train_size = train_size
    self.lower_case = lower_case
    self.verbose = verbose

    self.train_word_counts = {}
    self.word_counts = {}
    self.max_len = 0

    if snli_path_prefix ~= nil then
        self.verbose = false
        self.train = self:_load_data_file(snli_path_prefix .. "train.txt", self.train_word_counts)
        for k, v in pairs(self.train_word_counts) do self.word_counts[k] = v end
        self.dev = self:_load_data_file(snli_path_prefix .. "dev.txt", self.word_counts)

        self.verbose = verbose

        if self.train_size > 0 then
            self.train = tablex.sub(self.train, 1, self.train_size)
        end

        if self.verbose then
            printerr(string.format("SNLI train: %d pairs", #self.train))
            printerr(string.format("SNLI dev: %d pairs", #self.dev))
        end
    end

end


function SNLI:inc_word_counts(word, counter)
    if counter[word] ~= nil then
        counter[word] = counter[word] + 1
    else
        counter[word] = 1
    end
end


function SNLI:_load_data_file(file_path, word_counter)
    local data = {}
    for i, line in seq.enum(io.lines(file_path)) do
        local line_split = stringx.split(line, "\t")
        local gold_label = line_split[1]
        if self.relations[gold_label] ~= nil then
            if not pcall(
                function ()
                    -- Delete the punctuations in sentences, especially comma and dot.
                    local premise = stringx.split(string.sub(string.gsub(line_split[6], '%p', ''), 1, -1))
                    local hypothese = stringx.split(string.sub(string.gsub(line_split[7], '%p', ''), 1, -1))
                    if self.lower_case then
                        premise = moses.map(premise, function(i, v) return string.lower(v) end)
                        hypothese = moses.map(hypothese, function(i,v) return string.lower(v) end)
                    end

                    if #premise+1 > self.max_len then self.max_len = #premise + 1 end
                    if #hypothese+1 > self.max_len then self.max_len = #hypothese + 1 end
                    for i, v in ipairs(premise) do self:inc_word_counts(v, word_counter) end
                    for i, v in ipairs(hypothese) do self:inc_word_counts(v, word_counter) end

                    local ptree_str = stringx.join(" ", premise)
                    local htree_str = stringx.join(" ", hypothese)
                    -- local ptree = Tree:parse(ptree_str)
                    -- local htree = Tree:parse(htree_str)
                    data[#data+1] = {["label"] = self.relations[gold_label],
                        ["id"] = #data+1,
                        ["premise"] = ptree_str, ["hypothese"] = htree_str}
                end
            ) then
                if self.verbose then
                    printerr("error loading " .. line)
                end
            end
        end
    end
    return data
end

function SNLI:get_train_entail(dict)
    if self.train_entail == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.train do
            if self.train[i]['label'] == self.relations["entailment"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.train[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.train[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.train_entail = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.train_entail
end

function SNLI:get_train_neutral(dict)
    if self.train_neutral == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.train do
            if self.train[i]['label'] == self.relations["neutral"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.train[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.train[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.train_neutral = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.train_neutral
end

function SNLI:get_train_contradict(dict)
    if self.train_contradict == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.train do
            if self.train[i]['label'] == self.relations["contradiction"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.train[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.train[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.train_contradict = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.train_contradict
end

function SNLI:get_dev_entail(dict)
    if self.dev_entail == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.dev do
            if self.dev[i]['label'] == self.relations["entailment"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.dev[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.dev[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.dev_entail = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.dev_entail
end

function SNLI:get_dev_neutral(dict)
    if self.dev_neutral == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.dev do
            if self.dev[i]['label'] == self.relations["neutral"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.dev[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.dev[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.dev_neutral = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.dev_neutral
end

function SNLI:get_dev_contradict(dict)
    if self.dev_contradict == nil then
        local hypotheses = {}
        local premises = {}
        for i=1, #self.dev do
            if self.dev[i]['label'] == self.relations["contradiction"] then
                hypotheses[#hypotheses+1]=dict:convert(stringx.split(self.dev[i]['hypothese']))
                premises[#premises+1]=dict:convert(stringx.split(self.dev[i]['premise']))
            end
        end

        local t_hypotheses = torch.Tensor(#hypotheses, self.max_len):fill(0)
        local t_sos_premises = torch.Tensor(#premises, self.max_len):fill(0)
        local t_premises_eos = torch.Tensor(#premises, self.max_len):fill(0)
        for i=1, #hypotheses do
            t_hypotheses[i][{{self.max_len+1-hypotheses[i]:size(1), -1}}] = hypotheses[i]
            t_sos_premises[i][1] = dict:get_word_idx("<SOS>")
            t_sos_premises[i][{{2, premises[i]:size(1)+1}}] = premises[i]
            t_premises_eos[i][{{1, premises[i]:size(1)}}] = premises[i]
            t_premises_eos[i][premises[i]:size(1)+1] = dict:get_word_idx("<EOS>")
        end
        self.dev_contradict = {
            ["hypotheses"]=t_hypotheses,
            ["sos_premises"]=t_sos_premises,
            ["premises_eos"]=t_premises_eos
        }
    end
    return self.dev_contradict
end

function SNLI:get_batch_data(set, relation, dict, batch_index, batch_size, useGPU)
    if self.batch_data == nil then
        if set == 'train' then
            if relation == 'entail' then
                self.batch_data = self:get_train_entail(dict)
            end
            if relation == 'neutral' then
                self.batch_data = self:get_train_neutral(dict)
            end
            if relation == 'contradict' then
                 self.batch_data = self:get_train_contradict(dict)
            end
        end
        if set == 'dev' then
            if relation == 'entail' then
                self.batch_data = self:get_dev_entail(dict)
            end
            if relation == 'neutral' then
                self.batch_data = self:get_dev_neutral(dict)
            end
            if relation == 'contradict' then
                self.batch_data = self:get_dev_contradict(dict)
            end
        end
        if self.batch_data == nil then printerr('check out the name of dataset') end
        if useGPU then
            self.batch_data['hypotheses'] = self.batch_data['hypotheses']:cuda()
            self.batch_data['sos_premises'] = self.batch_data['sos_premises']:cuda()
            self.batch_data['premises_eos'] = self.batch_data['premises_eos']:cuda()
        end
    end
    return self.batch_data["hypotheses"][{{batch_index,
        math.min(batch_index + batch_size - 1, self.batch_data["hypotheses"]:size(1))}}]:t(),
    self.batch_data["sos_premises"][{{batch_index,
        math.min(batch_index + batch_size - 1, self.batch_data["sos_premises"]:size(1))}}]:t(),
    self.batch_data["premises_eos"][{{batch_index,
        math.min(batch_index + batch_size - 1, self.batch_data["premises_eos"]:size(1))}}]:t()

end

function SNLI:reset_batch_data()
    self.batch_data = nil
end

function SNLI:get_set_size(set, relation, dict)
    -- This function can return how many examples are contained in a certain dataset
    if set == 'train' then
        if relation == 'entail' then
            return self:get_train_entail(dict)["hypotheses"]:size(1)
        end
        if relation == 'neutral' then
            return self:get_train_neutral(dict)["hypotheses"]:size(1)
        end
        if relation == 'contradict' then
            return self:get_train_contradict(dict)["hypotheses"]:size(1)
        end
    end
    if set == 'dev' then
        if relation == 'entail' then
            return self:get_dev_entail(dict)['hypotheses']:size(1)
        end
        if relation == 'neutral' then
            return self:get_dev_neutral(dict)['hypotheses']:size(1)
        end
        if relation == 'contradict' then
            return self:get_dev_contradict(dict)['hypotheses']:size(1)
        end
    end
end