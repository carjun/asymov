import torch
import torch.utils.data as data

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
            dataset = data.TensorDataset(sequences)
            loader = data.DataLoader(dataset, batch_size=batch_size)
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