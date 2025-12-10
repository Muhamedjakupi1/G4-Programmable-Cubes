# Programmable-Cubes

## Përshkrimi i problemit
Ky projekt trajton optimizimin e montimit të një strukture hapësinore duke përdorur programmable cubes. Detyra konsiston në transformimin e një konfigurimi fillestar të kubeve në një formë të caktuar finale, duke përdorur vetëm lëvizje rrotulluese të lejuara dhe duke respektuar kufizime hapësinore dhe të numrit të komandave.

- Çdo kub mund të rrotullohet rreth kubeve fqinje.  
- Qëllimi është të gjendet sekuenca më efektive e lëvizjeve që e transformon strukturën sa më afër konfigurimit të synuar.
- Dallimi midis strukturës finale dhe objektivit matet dhe synohet të minimizohet.

## Instancat e problemit
Janë analizuar tre konfigurime me madhësi të ndryshme:

- **ISS (International Space Station)** – 148 kube, me limit 6000 komanda  
- **JWST (James Webb Space Telescope)** – 643 kube, me limit 30000 komanda  
- **Enterprise** – 1,472 kube, me limit 100000 komanda

## Metodat e zgjidhjes 
Në projekt janë përdorur tre qasje kryesore:

### **1. Baselines**
- **Random Search**: provon zgjidhje të rastësishme për të pasur një pikë krahasimi.
- 
### **2. Heuristics**
- **Greedy**: zgjedh lëvizjen më të favorshme në moment, por mund të ngecë në zgjidhje lokale.

### **3. Optimizers**

- **Algoritmi Gjenetik**: qasje evolucionare që përdor një grup zgjidhjesh dhe i përmirëson ato gradualisht përmes selektimit, kombinimit (crossover) dhe mutacionit.

- **Algoritmi Gjenetik i Avancuar**: version më i avancuar që punon me disa grupe paralelisht, ruan diversitetin e zgjidhjeve, mban një memorie të zgjidhjeve të mira dhe përdor strategji migrimi për të shmangur ngecjen në zgjidhje lokale.


## Struktura e projektit
```text
├── data/spoc3/cubes/          # Të dhënat e kubeve dhe konfigurimet ISS, JWST, Enterprise
├── problems/                  # Problem specification files
├── solver/                    # Algoritmet optimizuese
│   ├── baselines/             # Implementimi i random search 
│   ├── heuristics/            # Implementimi i greedy algorithm 
│   ├── optimizers/            # Implementimi i genetic algorithm 
│   └── results/               # Rezultatet dhe vizualizimet
├── src/                       # Komponentët kryesorë
│   ├── CubeMoveset.py         # Definimi i cube movement 
│   ├── programmable_cubes_UDP.py  # PyGMO problem interface
│   └── submission_helper.py   # Competition submission utilities
└── submissions/               # Files ku gjenerohen zgjidhjet
```


## Gjuha programuese e përdorur dhe libraritë e përdorura
  - **Python**  
- **NumPy** – për llogaritje numerike  
- **Numba** – për optimizim të performancës  
- **Matplotlib** – për vizualizime  
- **PyGMO** – për algoritmet e optimizimit


## Ekzekutimi i algoritmeve
Shembuj ekzekutimi për secilin skenar:

```bash
python solver/optimizers/iss/submit_ga_iss.py
python solver/optimizers/enterprise/submit_enhanced_ga_enterprise.py
python solver/heuristics/jwst/submit_greedy_jwst.py
```


