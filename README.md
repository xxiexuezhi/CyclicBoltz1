# CyclicBoltz1

CyclicBoltz1 is a cyclic peptide structure prediction and design pipeline built **on top of Boltz-1**.  
This repository provides scripts and configurations for running cyclic peptide predictions using the Boltz framework, with optional integration of ProteinMPNN for sequence-related components.

> This repository is **based on Boltz-1**.  
> It assumes you already have a working Boltz-1 installation and environment.

---

## Overview

- Built on **Boltz-1**
- Supports **cyclic peptide prediction and design**
- Uses **Boltz native prediction command**
- Compatible with **ProteinMPNN** (must be installed separately)
- No reimplementation of the Boltz inference engine — this repo focuses on configuration and usage

---

## Environment Setup

### 1. Install Boltz-1

First, install **Boltz-1** by following the official instructions in the Boltz repository. https://github.com/jwohlwend/boltz

Make sure:
- `boltz` command is available in your environment
- All Boltz dependencies are correctly installed (PyTorch, CUDA, etc.)

---

### 2. Install ProteinMPNN

ProteinMPNN is required for parts of the pipeline. 

Install it following the official ProteinMPNN repository instructions. https://github.com/dauparas/ProteinMPNN

Make sure:
- ProteinMPNN dependencies are installed
- You can successfully run ProteinMPNN scripts independently

---

### 3. Clone This Repository

```bash
git clone https://github.com/xxiexuezhi/CyclicBoltz1.git
cd CyclicBoltz1
