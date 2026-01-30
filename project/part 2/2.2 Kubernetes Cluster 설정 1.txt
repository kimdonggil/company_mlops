export K3S_NODE_NAME=k8s-cluster
export K3S_VERSION=1.29.5

sudo curl -sfL https://get.k3s.io | K3S_NODE_NAME=${K3S_NODE_NAME} INSTALL_K3S_VERSION=v${K3S_VERSION}+k3s1 sh -s - server \
  --docker \
  --write-kubeconfig-mode 644