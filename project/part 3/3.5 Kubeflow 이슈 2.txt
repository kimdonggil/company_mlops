export K3S_VERSION=v1.29.5+k3s1
sudo wget https://github.com/k3s-io/k3s/releases/download/${K3S_VERSION}/k3s-airgap-images-amd64.tar
sudo mkdir -p /var/lib/rancher/k3s/agent/images/
sudo cp ./k3s-airgap-images-amd64.tar /var/lib/rancher/k3s/agent/images/
sudo wget https://github.com/k3s-io/k3s/releases/download/${K3S_VERSION}/k3s
sudo cp ./k3s /usr/local/bin/
sudo chmod 755 /usr/local/bin/k3s
sudo curl -fsSL -o install.sh https://get.k3s.io/
sudo chmod +x install.sh
INSTALL_K3S_SKIP_DOWNLOAD=true ./install.sh \
  --write-kubeconfig-mode 644