# SMplice1.1
SMsplice1.1 is a hidden semi-Markov model based RNA splicing prediction model. It is an upgrade from the original [SMsplice model](https://github.com/kmccue/SMsplice) by allowing exonic and intronic SRE scores to be calculated from 3'SS and 5'SS separately, with additional options to use coding information to improve prediction performances.
## Dependencies
Install [awkde](https://github.com/mennthor/awkde)  
Clone this repository  
`git clone https://github.com/xnao25/SMsplice1.1.git`  
`cd SMsplice1.1`  
Prepare cython script  
`cythonize -3 -a -i SMspliceRFP.pyx`
## Run SMsplice1.1
### Command line options
python runSMspliceRFP.py 

#### Required parameters
- `-c` `--canonical_ss` The path to the canonical splicing site list. Choose one file from the directory `canonical_datasets` or create your own.
- `-a` `--all_ss` The path to all splicing site list. Choose one file from the directory `allSS_datasets` or create your own.
- `-g` `--genome` The path to reference genome FASTA file. 
- `-m` `--maxent_dir` The path to the trained MaxEnt model. Choose one folder in the directory `maxEnt_models`.

#### Optional parameters
- `-s` `--start_cds` The path to transcript start codon coordinates. Provide both
- `-st` `--stop_cds` The path to transcript stop codon coordinates.
- `-t` `--threads` Number of threads to run the training and scoring processes. This parameter is used to sort genes for multi-threading preparation. If not provided, multi-processing is still valid.
- `--use_reading_frame` To apply open reading frame counts of potential coding exons to splicing site predictions.
- `--use_periodic_pattern` To apply periodic patterns of SREs in coding exons to splicing site predictions.
- `-o` `--out_dir` The path to the output directory. If not provided, results will be created under the current directory.
- `-sp` `--save_parameters` To save the trained SRE scores in the output directory.
- `-si` `--split_intron` To train/apply intronic SREs at 3'SS and 5'SS separately. If this parameter is not used, intronic SREs at 3'SS and 5'SS will be combined to train.
- `-se` `--split_exon` To train/apply exonic SREs at 3'SS and 5'SS separately. If this parameter is not used, exonic SREs at 3'SS and 5'SS will be combined to train.
- `--prelearned_sres` The path to the directory of pre-trained SRE scores. Choose one folder in the directory `SRE_scores` or make your own files.
- `--learn_sres` To train new sets of SRE scores.
- `--max_learned_scores` The maximum learning counters. Default 1000.
- `--learning_seed` To initiate training set with decoy splicing sites or not. Use `real-decoy` or `none`. Default `none`.
- `--print_predictions` To print results on screen. If this parameter is not used, results will still be saved as a csv file.
- `--print_local_scores` Besides the prediction results, save the splicing site scores calculated using SREs as a csv file. 

