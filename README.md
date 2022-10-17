# Prototypical Networks for Few shot Learning in PyTorch

Predictive coding networks ([paper](https://arxiv.org/abs/2106.13082), [code](https://github.com/RobertRosenbaum/Torch2PC)) based implementation of Prototypical Networks for Few Shot Learning ([paper](https://arxiv.org/abs/1703.05175), [code](https://github.com/jakesnell/prototypical-networks)) in PyTorch. 
   
## Predictive Coding Algorithm

Inspired by the human brain, a predictive coding algorithm was introduced to resolve the biological limitation of backpropagation. Contrary to the neural plasticity of the human brain, the backpropagation algorithm performs global error-guided learning. However, in predictive coding, it performs local learning because its learning is performed with local error nodes in addition to the global error node. It has been demonstrated that an arbitrary computational graph can be trained in a predictive coding manner.

## Prototypical Networks

As shown in the reference paper Prototypical Networks are trained to embed samples features in a vectorial space, in particular, at each episode (iteration), a number of samples for a subset of classes are selected and sent through the model, for each subset of class `c` a number of samples' features (`n_support`) are used to guess the prototype (their barycentre coordinates in the vectorial space) for that class, so then the distances between the remaining `n_query` samples and their class barycentre can be minimized.

## Omniglot Dataset

Kudos to [@ludc](https://github.com/ludc) for his contribute: https://github.com/pytorch/vision/pull/46.
We will use the official dataset when it will be added to torchvision if it doesn't imply big changes to the code.

### Dataset splits

We implemented the Vynials splitting method as in [[Matching Networks for One Shot Learning](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)]. That sould be the same method used in the paper (in fact I download the split files from the "offical" [repo](https://github.com/jakesnell/prototypical-networks/tree/master/data/omniglot/splits/vinyals)). We then apply the same rotations there described. In this way we should be able to compare results obtained by running this code with results described in the reference paper.

## Prototypical Batch Sampler

As described in its PyDoc, this class is used to generate the indexes of each batch for a prototypical training algorithm.

In particular, the object is instantiated by passing the list of the labels for the dataset, the sampler infers then the total number of classes and creates a set of indexes for each class ni the dataset. At each episode the sampler selects `n_classes` random classes and returns a number (`n_support` + `n_query`) of samples indexes for each one of the selected classes.

## Prototypical Loss

Compute the loss as in the cited paper, mostly inspired by [this code](https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py) by one of its authors.

In [`prototypical_loss.py`](src/prototypical_loss.py) both loss function and loss class à la PyTorch are implemented. 

The function takes in input the batch input from the model, samples' ground truths and the number `n_suppport` of samples to be used as support samples. Episode classes get infered from the target list, `n_support` samples get randomly extracted for each class, their class barycentres get computed, as well as the distances of each remaining samples' embedding from each class barycentre and the probability of each sample of belonging to each episode class get finmally computed; then the loss is then computed from the wrong predictions probabilities (for the query samples) as usual in classification problems.

## Training with Backpropagation

Please note that the training code is here just for demonstration purposes. 

To train the Protonet on this task, cd into this repo's `src` root folder and execute:

    $ python train.py


The script takes the following command line options:

- `dataset_root`: the root directory where tha dataset is stored, default to `'../dataset'`

- `nepochs`: number of epochs to train for, default to `100`

- `learning_rate`: learning rate for the model, default to `0.001`

- `lr_scheduler_step`: StepLR learning rate scheduler step, default to `20`

- `lr_scheduler_gamma`: StepLR learning rate scheduler gamma, default to `0.5`

- `iterations`: number of episodes per epoch. default to `100`

- `classes_per_it_tr`: number of random classes per episode for training. default to `60`

- `num_support_tr`: number of samples per class to use as support for training. default to `5`

- `num_query_tr`: nnumber of samples per class to use as query for training. default to `5`

- `classes_per_it_val`: number of random classes per episode for validation. default to `5`

- `num_support_val`: number of samples per class to use as support for validation. default to `5`

- `num_query_val`: number of samples per class to use as query for validation. default to `15`

- `manual_seed`: input for the manual seeds initializations, default to `7`

- `cuda`: enables cuda (store `True`)

Running the command without arguments will train the models with the default hyperparamters values (producing results shown above).



## Training with Predictive Coding

Please note that the training code is here just for demonstration purposes. 

To train the predictive coding version of Protonet on this task with predictive coding manner, cd into this repo's `src` root folder and execute:

    $ python train_pc.py --error_type FixedPred --eta 0.1 --num_iter 20

- `error_type`: parameter update protocol of predictive coding algorithm, default to `FixedPred`

- `eta`: weight learning rate of predictive coding algorithm, default to `0.1`

- `num_iter`: the repetition number of backward iteration , default to `20`

The properties of other parameters are the same as backpropagation-based learning.


## .bib citation
cite the paper as follows (copied-pasted it from arxiv for you):
    
    @article{snell2017prototypical,
      title={Prototypical networks for few-shot learning},
      author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
      journal={Advances in neural information processing systems},
      volume={30},
      year={2017}
    }
    
    @article{rosenbaum2022relationship,
      title={On the relationship between predictive coding and backpropagation},
      author={Rosenbaum, Robert},
      journal={Plos one},
      volume={17},
      number={3},
      pages={e0266102},
      year={2022},
      publisher={Public Library of Science San Francisco, CA USA}
    }


## License

This project is licensed under the MIT License

Copyright (c) 2018 Daniele E. Ciriello, Orobix Srl (www.orobix.com).
