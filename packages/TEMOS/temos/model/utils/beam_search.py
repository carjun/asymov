import torch
import torch.utils.data as tud


def beam_search(model, sequences, predictions=20, beam_width=5, batch_size=16):
    """
    Note to Darsh: function implements Beam Search for diverse sequences in motion words
    (this one is simpler for sequence to sequence.
    Creating one with word maps as a better alternative, although it needs <start> and <end> ids for the tree.)

    The method can compute several outputs in parallel with the first dimension of sequences.
    Parameters
    ----------
    sequences: Tensor of shape (examples, length)
        The sequences to start the decoding process. (treating this as <start> for now.)
    predictions: int
        The number of tokens to stitch to sequences. Also the number of splits in the tree.
    beam_width: int
        The number of candidates to keep in the search.
    batch_size: int
        The batch size of the method.
    Returns
    -------
    sequences: Tensor of shape (examples, length + predictions)

    probabilities: FloatTensor of size examples
        The estimated log-probabilities for the output sequences.
    """
    with torch.no_grad():

        next_probabilities, next_latent, next_distribution = model.motion_to_motion_forward(sequences)[:, -1, :]
        #Using the motion_to_motion forward function from asymov

        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1) \
            .topk(k=beam_width, axis=-1)
        sequences = sequences.repeat((beam_width, 1, 1)).transpose(0, 1) \
            .flatten(end_dim=-2)
        next_chars = idx.reshape(-1, 1)
        sequences = torch.cat((sequences, next_chars), axis=-1)

        predictions_iterator = range(predictions - 1) #one prediction already done before for loop

        for i in predictions_iterator:
            dataset = tud.TensorDataset(sequences)
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            next_probabilities = []
            iterator = iter(loader)

            for (x,) in iterator:
                probabilities_i, latent_i, distribution_i = model.motion_to_motion_forward(x)[:, -1, :].log_softmax(-1)
                next_probabilities.append(probabilities_i)
            next_probabilities = torch.cat(next_probabilities, axis=0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, idx = probabilities.topk(
                k=beam_width,
                axis=-1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten() \
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(sequences.shape[0] // beam_width,device=sequences.device).unsqueeze(-1) * beam_width
            sequences = sequences[best_candidates].flatten(end_dim=-2)
            sequences = torch.cat((sequences, next_chars), axis=1)
        return sequences.reshape(-1, beam_width, sequences.shape[-1]), probabilities


def beam_search_nat(model, memory, beam_size, src_mask, max_len=256, start=0, end=1):
    assert beam_size > 1
    finished = torch.zeros(1, dtype=torch.bool)
    paths = torch.full((1, max_len + 1), start)
    probs = torch.zeros(1)

    for i in range(1, max_len + 1):
        mask = torch.triu(torch.ones((1, i,i)), diagonal=1)==0
        logits = model.decode(memory.expand((~finished).count_nonzero(), -1, -1),
            src_mask, paths[~finished, :i], mask)
        print(len(logits), len(logits[0]))
        scores = probs[~finished].unsqueeze(1) + model.generator(logits[:, -1])
        print(len(scores))
        if i == 1: # increase capacity to beam_size
            finished = finished.repeat(beam_size)
            paths = paths.repeat(beam_size, 1)
            probs = probs.repeat(beam_size)

        candidates = paths[~finished]
        topv, topi = torch.topk(scores.flatten(), beam_size)
        if any(finished): # length normalization
            for j in range(beam_size):
                finished[finished.nonzero(as_tuple=True)] ^= probs[finished] < (topv[j] / i)
            if (~finished).count_nonzero() > beam_size:
                beam_size = (~finished).sum()
                topv, topi = torch.topk(scores.flatten(), beam_size)

        paths[~finished] = candidates[
            torch.div(topi, model.tgt_vocab_size, rounding_mode='trunc')
        ]
        paths[~finished, i] = topi % model.tgt_vocab_size
        probs[~finished] = topv

        finished |= paths[:, i] == end
        beam_size = (~finished).count_nonzero()
        probs[paths[:, i] == end] /= i
        if all(finished): break

    best_path = paths[probs.argmax()]
    end_index = (best_path == end).nonzero()
    return best_path[1:end_index] if end_index.numel() else best_path[1:]


def beam_search_auto(
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    tgt_mask,
    tgt_padding_mask,
    end_symbol: int,
    max_len = 220,
    beam_width = 5,
    batch_size = 128            #CHECK: this batch size is different from the number of sequences passed
):

    with torch.no_grad():
        # pdb.set_trace()
        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]
        out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]
        logits = model.generator(out[-1])

        next_probabilities = logits#[-1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

        tgt = tgt.repeat((beam_width, 1))       #repeat BOS  for beam width
        tgt_padding_mask = tgt_padding_mask.unsqueeze(0).repeat((5,1,1))
        # next_chars = next_chars.reshape(-1, 1)
        # tgt = torch.cat((tgt, next_chars), axis=-1)
        tgt_len = tgt.new_full((beam_width, batch_size), max_len)#.repeat((beam_width, 1)) #[beam_width, Batch Size] #same dtype as tgt       
                                                                               #this needs to be sorted acc to best_canditates
        # pdb.set_trace()
        tgt = torch.cat((tgt.unsqueeze(-2), next_chars.transpose(1,0).unsqueeze(-2)), -2)      #concat next tokens for each beam (vertically)

        predictions_iterator = range(1, max_len - 1)        #1 (0th) prediction already done
        for i in predictions_iterator:
            # dataset = tud.TensorDataset(src.repeat((beam_width, 1, 1)), tgt)
            # loader = tud.DataLoader(dataset, batch_size=batch_size)

            tgt_mask = (T.generate_square_subsequent_mask(tgt.size(1))      #size(1) for num of decoded
                    .to(tgt.device, dtype=torch.bool))                      #tokens. 0th is beam size
            # pdb.set_trace()
            next_probabilities = torch.tensor([]) # will be containing probabs.(logits) for [batch,1004*beam_width]
            # iterator = iter(loader)
            # for x, y in iterator:

            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

            for b in range(beam_width):
                # memory_temp = model.encode(x[i], src_mask, src_padding_mask)
                # memory_temp = torch.concat([memory_temp, torch.zeros(memory_temp.shape[0], len(y[i])-memory_temp.shape[1], memory_temp.shape[2])], 1)
                # pdb.set_trace()
                out_temp = model.decode(tgt[b], memory, tgt_mask, None, tgt_padding_mask[b], src_padding_mask)
                logits_temp = model.generator(out_temp[-1])
                _, a = logits_temp.log_softmax(-1).topk(k=beam_width, axis=-1)
                # next_probabilities.append(logits_temp.squeeze().log_softmax(-1))
                next_probabilities = torch.cat((next_probabilities, logits_temp.squeeze()), axis=-1)      #probabilities [batch, beam*1004]
                # pdb.set_trace()

            # next_probabilities = torch.cat(next_probabilities, axis=0)
            # next_probabilities = next_probabilities.reshape((1, next_probabilities.shape[-1]))
            probabilities, idx = next_probabilities.log_softmax(-1).topk(k=beam_width, axis=-1)

            next_chars = torch.remainder(idx, vocabulary_size).transpose(1,0)#.flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()

            # pdb.set_trace()

            tgt_len = torch.where(torch.logical_and(next_chars==end_symbol, tgt_len==max_len), i+2, tgt_len)     #this requires debugging

            # tgt = tgt[best_candidates]#.flatten(end_dim=-2)

            for bc in range(len(best_candidates)): 
                tgt[:,:,bc] = tgt[:,:, bc][best_candidates[bc]]
            for bc in range(len(best_candidates)):                   #IMPROVEMENT: find better method to sort tgt_len on best_candidates
                tgt_len[:,bc] = tgt_len[:,bc][best_candidates[bc]]     #tgt_len is for each elemement in [beam, batch] for tgt, 
                                                                    #and arrangement of tgt keeps changing based on best_candidate beam

            # pdb.set_trace()           ##

            tgt = torch.cat((tgt, next_chars.unsqueeze(-2)), -2)

        return tgt



def diverse_beam_search_auto(                       #pre-alpha
    model,
    src,
    tgt,
    src_mask,
    src_padding_mask,
    tgt_mask,
    tgt_padding_mask,
    end_symbol: int,
    max_len = 220,
    beam_width = 5,
    batch_size = 128            #CHECK: this batch size is different from the number of sequences passed
):


    with torch.no_grad():

        memory = model.encode(src, src_mask, src_padding_mask)  # [Frames, Batches, *]
        out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)  # [Frames, Batch Size, *]
        logits = model.generator(out[-1])

        next_probabilities = logits#[-1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
        tgt = tgt.repeat((beam_width, 1))       #repeat BOS  for beam width
        tgt_padding_mask = tgt_padding_mask.unsqueeze(0).repeat((beam_width, 1, 1))
        # next_chars = next_chars.reshape(-1, 1)
        # tgt = torch.cat((tgt, next_chars), axis=-1)
        tgt_len = tgt.new_full((beam_width, batch_size), max_len)#.repeat((beam_width, 1)) #[beam_width, Batch Size] #same dtype as tgt
        # pdb.set_trace()
        tgt = torch.cat((tgt.unsqueeze(-2), next_chars.transpose(1,0).unsqueeze(-2)), -2)      #concat next tokens for each beam (vertically)
        
        # pdb.set_trace()               ##
        
        predictions_iterator = range(1, max_len - 1)     #1 (0th) prediction already done
        for i in predictions_iterator:
            tgt_mask = (T.generate_square_subsequent_mask(tgt.size(1))      #size(1) for num of decoded
                    .to(tgt.device, dtype=torch.bool))                      #tokens. 0th is beam size

            next_probabilities = torch.tensor([]) # will be containing probabs.(logits) for [batch,1004*beam_width]


            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(-1)], dim=-1)

            for b in range(beam_width):         #if using i, it's continued outside scope of for loop
                out_temp = model.decode(tgt[b], memory, tgt_mask, None, tgt_padding_mask[b], src_padding_mask)
                logits_temp = model.generator(out_temp[-1])

                if b == 0:
                  probabilities, idx = logits_temp.log_softmax(-1).topk(k=1, axis=-1)    ##
                  unique_tokens = idx           # needs to be debugged
                  torch.cat((tgt[b], unique_tokens.unsqueeze(-2)), -2)

                else:
                  probabilities, idx = logits_temp.log_softmax(-1).topk(k=b+1, axis=-1)

                  unique_mask = torch.stack([(tgt[:, -1, :] == idx.T[i]).any(dim=1) for i in range(idx.T.shape[0])], dim=1).long()
                  first_unique = unique_mask.argmin(1)
                  complete_index = torch.stack([torch.arange(unique_mask.shape[0]), first_unique], dim=1)   #containes coordinates -> diverse tokens, dim=0 -> batch_size

                  unique_tokens = idx[complete_index]       # needs to be debugged

                  torch.cat((tgt[b], unique_tokens.unsqueeze(-2)), -2)

                  torch.cat((tgt[b], unique_tokens.unsqueeze(-2)), -2)

            tgt_len = torch.where(torch.logical_and(next_chars==end_symbol, tgt_len==max_len), i+2, tgt_len)

        return tgt
