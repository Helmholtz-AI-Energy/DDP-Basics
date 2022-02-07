# Steps:
1. download torch image from NVIDIA (singularity)
	- `singularity pull new-torch-image-file.sif docker://nvcr.io/nvidia/pytorch:22.01-py3`
	- NOTE: if you are required to provide auth for the image: add this flag `--docker-login ` and enter your information when prompted
2. create a 'sandbox': `singularity build --sandbox --writable sandbox_torch/ torch.sif`
3. move to sandbox: `cd sandbox_torch/`
4. clone your git repo...i leave this to you
5. add the `comm.py` file to your repo and put it in a place that you can import the functions within
6. determine which `wireup-method` is best for you (this is system dependent) for HoreKa use `nccl-slurm-pmi`
7. when launching the job use `srun` and make sure to use the correct MPI when launching the jobs. for HoreKa use: `--mpi=pmi2`
8. export these two things WITHIN the `srun` call:
	- `MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);`
	- `MASTER_PORT=6000`
9. here is an example `train.sh` script which can be used to spawn a training
10. tell torch to use DDP training.
11. Make sure that the data and network on are the correct device. The local rank (if number of node-local ranks == number of node-local GPUs) can be found with `comm.local_rank()`
12. It may be useful to stop all ranks from printing / logging. get the global rank with `comm.get_rank()`

### Example Train script
```bash
#!/usr/bin/env bash

#SBATCH PARAMS GO HERE
#SBATCH --ntasks-per-node=4 # -> assuming that there are 4 GPUs / node


SRUN_PARAMS=(
  --mpi="pmi2"
  --label
)

ml purge

export DATA_DIR_TRAIN="/.../data/train"
export DATA_DIR_VAL="/.../data/test"
export CHECKPOINT_OUT="/.../checkpoints"
export CONTAINER="/.../containers/sandbox_torch/" 

export NCCL_IB_TIMEOUT=30
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

srun "${SRUN_PARAMS[@]}" singularity exec --nv --bind "/scratch","/tmp","${CHECKPOINT_OUT}","${DATA_DIR_TRAIN}","${DATA_DIR_VAL}"\
        ${CONTAINER} \
         bash -c "\
                MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
                MASTER_PORT=6000;
                python /your-repo-name/script.py \
                        -a qrnn3d \
                        --datadir ${DATA_DIR_TRAIN} \
                        --val-datadir ${DATA_DIR_VAL} \
                        --save-dir ${CHECKPOINT_OUT} \
                        --workers 4 \
                        --comm-method nccl-slurm-pmi \
                        --batch-size 2 \
                        --loss l2_ssim
                 "
```
### Code to tell torch to use DDP:
```
net = net.to(device)
net = nn.SyncBatchNorm.convert_sync_batchnorm(net, group)
net = nn.parallel.DistributedDataParallel(net, device_ids=[device.index])
```

## Notes:
- the sandbox image is not editable by the script. Files inside are read-only at runtime. All outputs should be to a different folder
- the `comm.py` is not foolproof. systems are different and may require special treatment.
- large nodes require a larger `timeout`  value for wireup. 
- the delay while creating the process groups is to avoid all ranks pinging a single rank and DDOS it
- 
