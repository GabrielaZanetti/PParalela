# CUDA & Simulação de Partículas — Exemplos Educacionais

## 📋 Visão Geral do Projeto

Este projeto implementa duas simulações computacionais intensivas — **conjunto de Mandelbrot** e **simulação de partículas** — em **CUDA** (GPU) e **Python** (CPU/NumPy).

### Objetivos Educacionais

1. **Computação Paralela em GPU**: Aprender conceitos de CUDA, kernels, threads, blocos e grids
2. **Otimização de Desempenho**: Comparar velocidade CPU (Python/NumPy) vs GPU (CUDA)
3. **Visualização Científica**: Gerar imagens e dados para análise
4. **Acessibilidade**: Fornecer versões executáveis sem exigir hardware GPU

---

## 🎨 Exemplo 1: Conjunto de Mandelbrot

### O que é?

O **conjunto de Mandelbrot** é um fractal definido por uma iteração simples no plano complexo:

$$z_{n+1} = z_n^2 + c$$

Para cada pixel $(x, y)$ da imagem:
- $c = x + iy$ (número complexo correspondente à posição)
- $z_0 = 0$
- Itera até $|z_n| > 2$ (diverge) ou atinge número máximo de iterações
- A **cor** representa quantas iterações foram necessárias

### Implementação

#### **mandelbrot.cu** (CUDA — GPU)
```cuda
__global__ void mandelbrot_kernel(unsigned char *img, int w, int h, int maxIter,
                                  double xmin, double xmax, double ymin, double ymax)
```

- **Parallelismo**: Cada thread do GPU calcula um pixel independentemente
- **Grid Layout**: 16×16 threads por bloco, múltiplos blocos cobrem toda a imagem
- **Saída**: Imagem PPM (P6 binary format) 24-bit RGB

#### **mandelbrot_python.py** (Python/NumPy — CPU)
- Loop Python simples com NumPy para operações vetorizadas
- Mais lento, mas portável (sem dependências de CUDA)

### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `width` | 1024 | Resolução horizontal em pixels |
| `height` | 768 | Resolução vertical em pixels |
| `maxIter` | 1000 | Máximo de iterações (mais = mais detalhes) |
| `output` | mandelbrot.ppm | Arquivo de saída |

### Exemplo de Saída

Arquivo PPM (formato texto): `P6\n{width} {height}\n255\n{RGB data}`
- Visualizável com qualquer viewer de imagem (GIMP, Windows Photo Viewer, etc.)
- Convertível para PNG/JPEG com ImageMagick ou PIL

---

## 🌌 Exemplo 2: Simulação de Partículas

### O que é?

Simula o movimento de **N partículas** em um campo de força central (atração ao centro).

#### Equações de Movimento

1. **Aceleração**: $\vec{a} = -G \frac{\vec{r}}{r^3 + \epsilon}$ (atração inversamente proporcional ao quadrado da distância)
2. **Velocidade**: $\vec{v}_{n+1} = \vec{v}_n + \vec{a} \cdot \Delta t$
3. **Posição**: $\vec{r}_{n+1} = \vec{r}_n + \vec{v}_n \cdot \Delta t$
4. **Amortecimento**: $\vec{v} \leftarrow \vec{v} \cdot \text{damping}$ (dissipação de energia)

### Implementação

#### **particles.cu** (CUDA — GPU)
```cuda
__global__ void integrate(Particle *p, int n, float dt, float damping, float G)
```

- **Estrutura**: `struct Particle { float3 pos; float3 vel; }`
- **Parallelismo**: 256 threads por bloco, cada thread integra uma partícula
- **Saída**: Arquivo CSV com posições e velocidades finais

#### **particles_python.py** (Python/NumPy — CPU)
- Vetorizado com NumPy: operações em arrays em vez de loops Python
- Simula a mesma física com mesmos resultados

### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `N` | 20000 | Número de partículas |
| `steps` | 200 | Número de passos de integração |
| `dt` | 0.01 | Tamanho do passo de tempo |
| `output` | particles.csv | Arquivo CSV de saída |

### Exemplo de Saída

Arquivo CSV:
```csv
id,x,y,z,vx,vy,vz
0,-0.234567,0.145678,-0.987654,0.012345,-0.056789,0.001234
1,0.456789,-0.234567,0.123456,-0.045678,0.023456,0.012345
...
```

Pode ser visualizado com:
- Excel/LibreOffice Calc
- Python (Pandas + Matplotlib)
- Script `visualize.py` (gráfico 3D interativo)

---

## 🚀 Como Usar

### Opção 1: Python (Recomendado para iniciar)

Não requer compilação. Rápido de testar.

#### Instalação

```powershell
# Instalar dependências
pip install numpy matplotlib
```

#### Executar Exemplos

```powershell
cd c:\Users\gabii\GitHub\PParalela\trabalho3\cuda_examples

# Mandelbrot (512×384, 256 iterações) — ~1-2 segundos
python mandelbrot_python.py 512 384 256 mandelbrot_quick.ppm

# Mandelbrot detalhado (1920×1080, 1000 iterações) — ~30 segundos
python mandelbrot_python.py 1920 1080 1000 mandelbrot_hd.ppm

# Partículas (1000 partículas, 100 passos) — ~1 segundo
python particles_python.py 1000 100 0.01 particles_small.csv

# Partículas grandes (20000 partículas, 500 passos) — ~20 segundos
python particles_python.py 20000 500 0.01 particles_large.csv
```

#### Executar Demonstração Completa

```powershell
python demo.py
```

Gera:
- `mandelbrot_demo.ppm` (1024×768)
- `particles_demo.csv` (10000 partículas)

#### Visualizar Resultados

```powershell
# Visualizar Mandelbrot com matplotlib
python visualize.py mandelbrot_demo.ppm

# Visualizar partículas em 3D
python visualize.py particles_demo.csv

# Visualizar arquivo PPM manualmente
# Windows: clique 2x no arquivo .ppm
# Linux/Mac: feh, eog, ou outro viewer de imagem
```

---

### Opção 2: CUDA (GPU — Muito Mais Rápido)

Requer NVIDIA GPU + CUDA Toolkit instalado.

#### Instalação do CUDA Toolkit

**Em WSL (Ubuntu)**:
```bash
wsl -d Ubuntu
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit gcc g++ make
```

**No Windows nativo**:
1. Instalar [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Instalar [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Adicionar `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin` ao PATH

#### Compilar

**Em WSL**:
```bash
cd /mnt/c/Users/gabii/GitHub/PParalela/trabalho3/cuda_examples
make clean && make
```

**No Windows**:
```powershell
cd c:\Users\gabii\GitHub\PParalela\trabalho3\cuda_examples

# Compilação manual
nvcc -O3 mandelbrot.cu -o mandelbrot.exe
nvcc -O3 particles.cu -o particles.exe

# Ou usar make (se instalado)
make
```

#### Executar

```bash
# Mandelbrot
./mandelbrot 1920 1080 1000 mandelbrot_gpu.ppm

# Partículas
./particles 20000 500 0.01 particles_gpu.csv
```

---

## 📊 Estrutura de Arquivos

```
trabalho3/cuda_examples/
├── README.md                    # Este arquivo
├── Makefile                     # Compilação CUDA
│
├── mandelbrot.cu                # Kernel CUDA — Mandelbrot
├── mandelbrot_python.py         # Implementação Python
├── particles.cu                 # Kernel CUDA — Partículas
├── particles_python.py          # Implementação Python
│
├── visualize.py                 # Visualizador (PPM + CSV)
├── demo.py                      # Script de demonstração
│
├── mandelbrot_demo.ppm          # Exemplo de saída (imagem)
├── particles_demo.csv           # Exemplo de saída (dados)
└── ...outros .ppm e .csv...     # Outputs gerados
```

---

## 🔬 Análise de Desempenho

Comparação esperada (em máquinas modernas):

| Tarefa | Python (CPU/NumPy) | CUDA (GPU) | Aceleração |
|--------|-------------------|-----------|-----------|
| Mandelbrot 512×384, 256 iter | ~1-2s | ~10ms | **100-200×** |
| Mandelbrot 1920×1080, 1000 iter | ~30-40s | ~50ms | **600-800×** |
| Partículas 10k, 500 steps | ~5-10s | ~50ms | **100-200×** |

> **Nota**: A aceleração CUDA é significativamente maior porque ambas as tarefas são:
> - **Embarrassingly parallel**: cada pixel/partícula é independente
> - **Computacionalmente intensiva**: muito trabalho por dado
> - **Memory-bound em CPU**: NumPy mesmo otimizado não compete com GPU

---

## 🛠️ Detalhes Técnicos

### CUDA Concepts

- **Thread**: Unidade mínima de execução (~1000s rodam em paralelo)
- **Block**: Conjunto de threads que compartilham memória compartilhada (até 1024)
- **Grid**: Conjunto de blocos que cobrem toda a computação
- **Kernel**: Função executada no GPU

Exemplo de configuração:
```cuda
dim3 block(16, 16);              // 256 threads (2D grid de pixels)
dim3 grid((w+15)/16, (h+15)/16); // Múltiplos blocos para cobrir imagem
mandelbrot_kernel<<<grid, block>>>(d_img, w, h, ...);
```

### Formatos de Saída

**PPM (Netpbm)**:
- Simples, sem compressão, fácil de escrever
- Lê-se: `P6\n<width> <height>\n255\n<RGB bytes>`
- Aberto por qualquer viewer de imagem

**CSV**:
- Texto puro, facilmente importável
- Colunas: `id, x, y, z, vx, vy, vz`
- Compatível com Excel, Python Pandas, etc.

---

## 📚 Referências & Leitura Adicional

1. **Mandelbrot Set**
   - Wikipedia: https://en.wikipedia.org/wiki/Mandelbrot_set
   - Matemática: https://mathworld.wolfram.com/MandelbrotSet.html

2. **CUDA Programming**
   - NVIDIA CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
   - Beginner's Guide: https://developer.nvidia.com/cuda-toolkit

3. **Integração Numérica**
   - Método de Euler (usado aqui)
   - Métodos de Runge-Kutta (mais precisos)

4. **Visualização em Python**
   - Matplotlib: https://matplotlib.org/
   - NumPy: https://numpy.org/

---

## ❓ FAQ

**P: Posso rodar CUDA sem NVIDIA GPU?**
R: Não, você precisa de uma placa NVIDIA. Use Python como alternativa.

**P: O arquivo PPM é muito grande, posso comprimir?**
R: Sim, converta com ImageMagick: `magick mandelbrot.ppm mandelbrot.png`

**P: Como mudo a região do Mandelbrot renderizada?**
R: Modifique `xmin, xmax, ymin, ymax` no código-fonte (padrão: -2, 1, -1.2, 1.2).

**P: Posso rodar 100k partículas?**
R: Sim, mas vai levar tempo. Python: ~1-2 min. CUDA: ~100ms.

**P: Qual GPU recomenda?**
R: Qualquer NVIDIA recente (RTX 3060+, RTX 4090, A100). Até GTX 1050 funciona.

---

## 📝 Licença

Código educacional. Livre para usar, modificar e distribuir.
