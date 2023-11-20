#   OM2Seq

Code for the paper: [OM2Seq: Learning retrieval embeddings for optical genome mapping]()

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
