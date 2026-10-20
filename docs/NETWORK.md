# Network: TP=2 over the ConnectX-7 link

The two Sparks run one model with tensor parallelism (TP=2). Every layer exchanges activations between the GPUs, so the link between the nodes decides the speed. This recipe uses the QSFP cable between the two ConnectX-7 (CX7) ports, with RoCE v2 (RDMA over Ethernet).

## What you need

- **Cable.** One QSFP cable between the Sparks (the setup in NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) playbook; do it first). The recipe was measured with both CX7 ports active.
- **Addresses.** Static IPv4 addresses on the CX7 netdev, one subnet, for example `10.100.80.1/24` (head) and `10.100.80.2/24` (worker). Put them in `.env` as `HEAD_IP` and `WORKER_IP`.
- **ssh.** Key-based ssh from the head to the worker (`WORKER_SSH`). `docker/build.sh`, `start.sh` and `stop.sh` use it.
- **RDMA devices.** The RDMA device names and the netdev name for `.env`:

```bash
ibdev2netdev
```

```bash
ip -br addr
```

  Example from the measured pair: `rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)` and `roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)`. Then:
  - `RDMA_DEVICES` and `NCCL_IB_HCA` = both RoCE devices;
  - `NCCL_SOCKET_IFNAME` = the netdev that holds `HEAD_IP`.

  If `ibdev2netdev` is not installed, this shows the same pairs (`<rdma device>: <netdev>`):

```bash
for d in /sys/class/infiniband/*; do echo "$(basename "$d"): $(ls "$d/device/net")"; done
```

- **MTU 9000** on each CX7 netdev, on both nodes. See [MTU](#mtu).

## MTU

The engine sends its RDMA gather in 4096-byte packets. RoCE uses MTU 4096 only when the netdev MTU is about 4200 or more. NVIDIA's playbook leaves the netdev at the default MTU 1500. Then RoCE uses MTU 1024, the packets drop, and both ranks stop in the warm-up prefill with `rdma gather: RDMA completion: transport retry counter exceeded`. `ping` and NCCL still work, so the fault looks like a bad cable. The measured pair used MTU 9000.

`scripts/preflight.sh` checks this. To fix it:

1. Set MTU 9000 on each CX7 netdev, on both nodes. Use your netdev names from `ip -br addr`:

```bash
sudo ip link set enp1s0f1np1 mtu 9000
```

2. Make it persistent. Add `mtu: 9000` to each CX7 netdev in your netplan file (for example `/etc/netplan/40-cx7.yaml`), then apply it:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      mtu: 9000
```

```bash
sudo netplan apply
```

3. Check that a 9000-byte frame passes with no fragmentation. Run this on the head:

```bash
ping -c 3 -M do -s 8972 10.100.80.2
```

4. Check that RoCE now uses MTU 4096:

```bash
ibv_devinfo -d rocep1s0f1 -i 1 | grep active_mtu
```

## The GID index

RoCE v2 over IPv4 uses one entry of the port's GID table. On the measured pair it was index 3 (`GID_INDEX=3`).

**Trap:** after a link flap (a cable reseat, or one node rebooting), the GID tables on the two nodes can shift, and a fixed index then points at different entries on each side. NCCL init then hangs or fails. If that happens:

1. Look at the table on both nodes:

```bash
show_gids
```

2. Set `GID_INDEX` to the RoCE v2 IPv4 entry on both nodes, or set `GID_INDEX=-1` to let NCCL choose.
3. Restart:

```bash
./stop.sh && ./start.sh
```

## Check the link before you start

```bash
ping -c 3 -I enp1s0f1np1 10.100.80.2
```

```bash
ib_write_bw -d rocep1s0f1 -x 3 10.100.80.1
```

Run `ib_write_bw -d rocep1s0f1 -x 3` (no address) on the head first, then the command above on the worker. A healthy link shows tens of GB/s.

## Ports used

| Port | Node | What |
|---|---|---|
| 29551/tcp | head | TensorFold rendezvous store (rank 1 connects to it) |
| ephemeral tcp | head | TensorFold step link and watchdog heartbeats (rank 0 to rank 1) |
| `PORT` (8888) | head | OpenAI-compatible API (rank 0 only) |
