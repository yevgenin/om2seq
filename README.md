#   OM2Seq

Code for the paper: [OM2Seq: Learning retrieval embeddings for optical genome mapping]()

The dataset used in this paper is available as a parquet file at: [https://zenodo.org/records/10160960/files/dataset.parquet](https://zenodo.org/records/10160960/files/dataset.parquet)

#   Setup shell environment

Before running any of the commands below, you need to setup the shell environment:

```shell
. run.sh
```

#   Building the Docker container

```shell
docker-build
```


#   Training the model

```shell
train
docker-run-gpu
```

#   Running the Benchmark

```
benchmark
docker-run-gpu
```

#   Reproducing the figures from the paper

```shell
figures-all
docker-run
```
