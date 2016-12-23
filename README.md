# Learning Sparse Low Rank Tensor Representations of Transitive Verbs

This is a recreation and modification of

```
Fried, D., Polajnar, T., & Clark, S. (2015). Low-rank tensors for verbs in compositional distributional semantics. In Proceedings of the 53nd Annual Meeting of the Association for Computational Linguistics (ACL 2015), Bejing, China.
Chicago
```


---


Files
-----
* `skicka_upload.sh`

shell script to quickly upload output logs and saved weights to Google Drive using Skicka

* `slurm_run.sbatch`
* `slurm_test.sbatch`

scripts to run `train_parallel.py` on cores using SLURM job scheduler

* `train_parallel.py`
* `train_verbs.py`

python scripts to train verb tensors with and without OpenMPI parallelization, respectively

* `utils.py`

simple utility functions used in `verb.py` and training scripts

* `verb.py`

contains `Verb` class, includes implementations of SGD and ADADELTA to train verb tensors

---


Requires
--------
- Numpy
- Pandas
- tqdm
- [SimpleMPI](https://github.com/piantado/SimpleMPI/tree/master/SimpleMPI) (for `train_parallel.py`)



---



Full report will be available soon.

