#!/bin/bash


# Step 1: cgroup
if [ $# -eq 1 ]; then
  echo "++ Configuring cgroup to simulate a slower CPU clock speed..."
  mkdir -p /sys/fs/cgroup/system_limit
  CPU_PERCENTAGE=$1
  echo "${CPU_PERCENTAGE}000" > /sys/fs/cgroup/system_limit/cpu.max
  echo $(pidof bash | awk '{print $NF}') > /sys/fs/cgroup/system_limit/cgroup.procs
fi

# Step 2: socket buffers
echo "++ Changing maximum socket buffer sizes..."
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.wmem_max=13631488

# Step 3: network shaping
echo "++ Executing shape.sh (with net_rules.tsv) to simulate bandwidth shaping..."
./shape.sh net_rules.tsv
