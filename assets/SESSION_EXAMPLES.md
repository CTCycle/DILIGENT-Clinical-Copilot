# SESSION_EXAMPLES

Questo file contiene 3 casi completi, pronti per essere copiati/incollati nei campi della pagina **Drug-Induced Liver Injury analysis**.

Campi richiesti dalla UI:
- Patient Name
- Visit Date
- Anamnesis
- Current Drugs
- Laboratory Analysis

Nota: i casi sono **anonimizzati** e orientati al test funzionale.

---

## CASE 1 - Pattern epatocellulare (sospetto da fitoterapici + PPI)

### Patient Name
Mario R.

### Visit Date
2026-03-14

### Retrieval Controls
- Internal RAG: OFF
- Web Search: OFF

### Anamnesis
Paziente maschio, 58 anni. Anamnesi di MRGE con esofago di Barrett in terapia cronica con inibitore di pompa protonica. Da circa 7-10 giorni nausea, astenia, urine ipercromiche e comparsa di ittero. Nega abuso alcolico, nega uso recente di paracetamolo ad alte dosi, nega esposizioni professionali note.

Negli ultimi 2 mesi ha assunto prodotti da banco/erboristici a scopo antalgico (boswellia, bromelina, miele di manuka). Nessun viaggio extraeuropeo recente. Sierologie virali già avviate dal curante con esito preliminare negativo per epatiti virali principali.

Quadro clinico compatibile con epatite acuta con componente colestatica associata. In miglioramento parziale dopo sospensione dei prodotti erboristici.

### Current Drugs
- Esomeprazolo 40 mg PO 1 volta/die, terapia cronica (>12 mesi)
- Boswellia serrata estratto secco, 1 cps BID, iniziata circa 6 settimane prima dell'ittero, sospesa alla comparsa sintomi
- Bromelina 500 mg PO BID, iniziata circa 5 settimane prima dell'ittero, sospesa alla comparsa sintomi
- Miele di Manuka 1-2 cucchiaini/die, iniziato circa 4 settimane prima dell'ittero, sospeso

Farmaci non assunti: paracetamolo ad alto dosaggio, antibiotici recenti, antifungini sistemici.

### Laboratory Analysis
Data prelievo principale: 2026-03-04
- Bilirubina totale: 227 umol/L
- ALT: 2434 U/L
- AST: 1321 U/L
- ALP: 111 U/L
- GGT: 472 U/L
- INR: 1.1
- Lipasi: 27 U/L

Follow-up 2026-03-15
- Bilirubina totale: 145 umol/L
- ALT: 885 U/L
- AST: 457 U/L
- ALP: 147 U/L
- GGT: 258 U/L

Sierologie/autoimmunita (riassunto):
- HAV/HBV/HCV/HEV: negative (in prima valutazione)
- EBV/CMV: negative
- ANA borderline (1:160)

---

## CASE 2 - Pattern colestatico/misto (paziente oncologica in politerapia)

### Patient Name
Elena B.

### Visit Date
2026-02-18

### Retrieval Controls
- Internal RAG: ON
- Web Search: ON

### Anamnesis
Paziente femmina, 60 anni, adenocarcinoma polmonare in trattamento oncologico. Ricoverata per incremento progressivo degli indici di colestasi e prurito diffuso, senza febbre. Riferisce iporessia e calo ponderale moderato nelle ultime settimane. Nessuna storia di epatopatia cronica nota.

In corso terapia antineoplastica e profilassi antiinfettiva. Temporale sospetto: peggioramento laboratoristico dopo ultima somministrazione immunoterapia/combinazione sistemica. Ecografia epatobiliare senza chiara ostruzione meccanica significativa.

Necessaria valutazione di nesso causale tra farmaci oncologici concomitanti e danno epatico farmaco-indotto.

### Current Drugs
- Nivolumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
- Ipilimumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
- Trastuzumab deruxtecan EV (linea precedente, sospeso 6 settimane fa)
- Cotrimossazolo 800/160 mg PO 3 volte/settimana, profilassi
- Pantoprazolo 40 mg PO/die
- Ondansetron 8 mg PRN nausea

### Laboratory Analysis
Data prelievo principale: 2026-02-16
- Bilirubina totale: 98 umol/L
- Bilirubina diretta: 71 umol/L
- ALT: 412 U/L
- AST: 355 U/L
- ALP: 684 U/L
- GGT: 912 U/L
- INR: 1.2
- Albumina: 33 g/L

Follow-up 2026-02-21
- Bilirubina totale: 121 umol/L
- ALT: 390 U/L
- AST: 301 U/L
- ALP: 702 U/L
- GGT: 980 U/L

Altri elementi:
- HBV/HCV: negativi
- Autoimmunita: non conclusiva
- Imaging: no dilatazione vie biliari significativa

---

## CASE 3 - Anziana fragile con antibiotico recente (pattern misto, DD con sepsi/ischemia)

### Patient Name
Giuseppina P.

### Visit Date
2026-02-20

### Retrieval Controls
- Internal RAG: ON
- Web Search: OFF

### Anamnesis
Paziente femmina, 84 anni, IRC stadio 3b, ipertensione arteriosa, scompenso cardiaco cronico, diabete tipo 2. Ricoverata per infezione urinaria complicata trattata con antibiotico EV, poi transizione orale.

Dopo 5-7 giorni di terapia antibiotica compare incremento di transaminasi e bilirubina con modesto peggioramento clinico generale (astenia, iporexia, nausea lieve, assenza di dolore addominale importante). Nessun consumo alcolico. Nessuna epatopatia cronica documentata.

Quadro da discutere in diagnosi differenziale con epatopatia da farmaco vs danno epatico associato a infezione sistemica/ipoperfusione.

### Current Drugs
- Piperacillina/tazobactam 4.5 g EV q8h, iniziata 2026-02-10, sospesa 2026-02-16
- Amoxicillina/acido clavulanico 875/125 mg PO BID, iniziata 2026-02-16
- Furosemide 25 mg PO/die
- Bisoprololo 2.5 mg PO/die
- Amlodipina 5 mg PO/die
- Insulina basal-bolus secondo schema interno
- Atorvastatina 20 mg PO/die (cronica)

### Laboratory Analysis
Baseline pre-ricovero (2026-01-25)
- Bilirubina totale: 14 umol/L
- ALT: 32 U/L
- AST: 28 U/L
- ALP: 118 U/L

Picco durante ricovero (2026-02-17)
- Bilirubina totale: 76 umol/L
- ALT: 286 U/L
- AST: 244 U/L
- ALP: 356 U/L
- GGT: 421 U/L
- INR: 1.3
- Creatinina: 176 umol/L
- PCR: 88 mg/L

Follow-up (2026-02-22)
- Bilirubina totale: 63 umol/L
- ALT: 210 U/L
- AST: 162 U/L
- ALP: 330 U/L
- GGT: 390 U/L

Microbiologia/virologia:
- Emocolture negative
- HBsAg, anti-HCV: negativi



Anamnesi--------
Adenocarcinoma polmonare del segmento apico-dorsale del lobo
superiore sinistro, stadio clinico cT1b-3 cN1 MX, diagnosi 16.03.2023
Carcinoma duttale in situ (ER-, PR- G3 di tipo comedonico e
cribriforme) radioterapia adiuvante il 03.04.2012
Carcinoma lobulare invasivo del seno sinistro stadio pT1c (m), pN0 sn
0/2 10.12.2018 [ER+, PR, c-erbB-2 score 2] ormonoterapia adiuvante
con Letrozolo terminata il 14.05.2019 seguita da Tamoxifene dal
15.05.2019 al 10.10.2020
Sindrome coronarica acuta tipo STEMI anteriore (06.01.2024) su
malattia trivasale
Insufficienza cardiaca con FE ridotta (HFrEF) su base ischemica
Infezione delle basse vie respiratorie da Influenza A con sospetta
sovrainfezione batterica
Asma bronchiale eosinofila, non escludibile BPCO
Ipertensione arteriosa in trattamento
Malnutrizione proteico-energetica di grado moderato (Score NRS 4) 

Laboratorio
 
Labor 10.01.2025: ALAT 345 U/L, ALP 1055 U/L, Amilasi P 94 U/L, Lipasi 172 U/L.
Labor 16.01.2025: ALAT 822 U/L, ASAT 344 U/L, ALP 898 U/L, Amilasi P 119 U/L, Lipasi 215 U/L.
Labor 13.02.2025: ALAT 591 U/L, ASAT 205 U/L, ALP 666 U/L, Amilasi P 89 U/L, Lipasi 145 U/L.
Riferitidall'oncologoCurantevalorialteratigiàapartiredal17.12.2024(precedentidinovembrenella
norma),percuièstatasospesaterapiaantibioticainattoconInvanz®(Ertapenem),chelapaziente
aveva iniziato i primi giorni di dicembre 2024, senza risoluzione del quadro ma in peggioramento.
 
CT del 07.01.2025 -Fegato: nei limitimorfovolumetrici dellanorma, esentedaalterazioni densitometrichesospette.
Cisti epatiche stabili. -Colecistieviebiliari: fisiologicamentedistesa,esentedalitiasidievidenzaTC.Nondilatatelevie
biliari intraepatiche e la via biliare principale. -Pancreas: regolare per volume emorfologia, senza alterazioni densitometriche focali. Non
dilatato il dotto pancreatico principale.

Terapia

■Amlodipin axapharm cpr 5 mg 0-0-1-0 per os
■Bilol cpr rivestite 5 mg 1-0-0-0 per os
■Clopidogrel Spirig HC cpr rivestite 75 mg 1-0-0-0 per os
■Aspirin Cardio 100 mg cpr [cpr] 1-0-0-0 per os
■Pantoprazol Helvepharm cpr rivestite 20 mg 1-0-0-0 per os
■Prednison 20 mg cpr [cpr] 2-0-0-0 per os
 Dal 15.01.2025 40 mg (inizio terapia il 6-7 gennaio, alla dose di 60 mg/die) - Peso della paziente 
51.60 kg
 
■Seresta 15 mg cpr [cpr] 0-0-0-0.5 per os
■Venlafaxin ER Sandoz Ret caps 75 mg 1-0-0-0 per os
■Forxiga 10 mg cpr [cpr] 1-0-0-0 per os
■Diovan 80 mg cpr [cpr] 0-0-0-0 per os
 se PAS>o= 100 mmHg 
■Domperidon axapharm lingual cpr orodisp 10 mg 0-0-0-0 per os
 In riserva: se nausea, vomito 
■Seresta 15 mg cpr [cpr] 0-0-0-0 per os
 In riserva: se insonnia 
■ Invanz sol iniet [g] 0-0-0-0 i.