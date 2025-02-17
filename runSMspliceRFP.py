import numpy as np
import pandas as pd
import time, argparse  # , json, pickle
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
import pickle
from SMspliceRFP import *
# import SMspliceC as SMSpliceC
import os, sys

startTime = time.time()


# SMSplice, original, RF feature ,and periodic patterns

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="parsing arguments")
    parser.add_argument("-c", "--canonical_ss", required=True)
    parser.add_argument("-a", "--all_ss", required=True)
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-m", "--maxent_dir", required=True)
    parser.add_argument("-s", "--start_cds", default='none')  # get information about the position of the start codon
    parser.add_argument("-st", "--stop_cds", default='none')  # get the coordinates of stop codons
    parser.add_argument("-t", "--threads", default=0, type=int)
    parser.add_argument("--use_reading_frame", action='store_true')  # choices=["True","False"],default="False")
    parser.add_argument("--use_periodic_pattern", action='store_true')  # choices=["True","False"],default="False")
    parser.add_argument("-o", "--out_dir", default=False)  # output directory
    parser.add_argument("-sp", "--save_parameters", action='store_true')  # ,choices=['Y','N'],default='N')

    parser.add_argument("-si", "--split_intron", action='store_true')  # choices=['Y', 'N'], default='N')
    parser.add_argument("-se", "--split_exon", action='store_true')  # choices=['Y', 'N'], default='N')

    parser.add_argument("--prelearned_sres",
                        default='none')  # , choices = ['none', 'human', 'mouse', 'zebrafish', 'fly', 'moth', 'arabidopsis'], default = 'none')
    parser.add_argument("--learn_sres", action="store_true")
    parser.add_argument("--max_learned_scores", type=int, default=1000)

    parser.add_argument("--learning_seed", choices=['none', 'real-decoy'], default='none')
    parser.add_argument("--max_learned_weights", type=int, default=15)

    parser.add_argument("--print_predictions", action="store_true")
    parser.add_argument("--print_local_scores", action="store_true")

    opts = parser.parse_args()
    return opts


args = parse_arguments()
np.random.seed(25)

# Prepare output folders and data frames
if args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
else:
    args.out_dir = os.getcwd()

pickle_path = os.path.join(args.out_dir, 'update_steps')
os.makedirs(pickle_path, exist_ok=True)

# output = os.path.join(directory_path, "results.txt")
performance = []
# Load data from arguments
maxEntDir = args.maxent_dir
genome = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
canonical = pd.read_csv(args.canonical_ss, sep='\t', engine='python', index_col=0, header=None)

if args.start_cds != 'none' and args.stop_cds != 'none':
    cds_start = pd.read_csv(args.start_cds, sep='\t', names=['chr', 'position', 'strand', 'gene'])
    cds_start['gene'] = cds_start['gene'].apply(lambda x: x.replace('"', '').replace(';', ''))
    cds_start.index = cds_start['gene']
    cds_end = pd.read_csv(args.stop_cds, sep='\t', names=['chr', 'position', 'strand', 'gene'])
    cds_end['gene'] = cds_end['gene'].apply(lambda x: x.replace('"', '').replace(';', ''))
    cds_end.index = cds_end['gene']
else:
    cds_start = pd.read_csv(args.canonical_ss, sep='\t',
                            names=['gene', '1', 'chr', 'strand', 'position1', 'position2', '3', '4'])
    cds_end = pd.read_csv(args.canonical_ss, sep='\t',
                          names=['gene', '1', 'chr', 'strand', 'position1', 'position2', '3', '4'])
    cds_start['position'] = cds_start.apply(lambda x: x['position1'] if x['strand'] == '+' else x['position2'], 1)
    cds_end['position'] = cds_end.apply(lambda x: x['position2'] if x['strand'] == '+' else x['position1'], 1)
    cds_start['gene'] = cds_start['gene'].apply(lambda x: x.replace('"', '').replace(';', ''))
    cds_start.index = cds_start['gene']
    cds_end['gene'] = cds_end['gene'].apply(lambda x: x.replace('"', '').replace(';', ''))
    cds_end.index = cds_end['gene']
if args.start_cds != 'none' and args.stop_cds != 'none':
    canonical = canonical.loc[[x for x in canonical.index if x in list(cds_start['gene'])]]
canonical.index = canonical.index.map(str)
allSS = pd.read_csv(args.all_ss, sep='\t', engine='python', index_col=0, header=None)
allSS.index = allSS.index.map(str)
canonical[2] = ['chr' + x if 'chr' not in x else x for x in canonical[2]]
genes = {}
start_stop = {}
for gene in canonical.index:
    txstart = canonical.loc[gene, 4] - 1
    txend = canonical.loc[gene, 5]
    chrom = canonical.loc[gene, 2][3:] if 'TAIR10' in args.canonical_ss else canonical.loc[gene, 2]
    if not np.array_equiv(['a', 'c', 'g', 't'],
                          np.unique(genome[chrom][canonical.loc[gene, 4] - 1:canonical.loc[gene, 5]].seq.lower())):
        continue

    name = gene
    geneID = gene
    description = gene + ' GeneID:' + gene + ' TranscriptID:Canonical' + ' Chromosome:' + canonical.loc[gene, 2] + \
                  ' Start:' + str(txstart) + ' Stop:' + str(txend) + ' Strand:' + canonical.loc[gene, 3]

    if canonical.loc[gene, 3] == '-':
        seq = genome[chrom][canonical.loc[gene, 4] - 1:canonical.loc[gene, 5]].seq.reverse_complement()
        if args.start_cds != 'none' and args.stop_cds != 'none':
            start_stop[gene] = [txend - cds_start.loc[gene, 'position'], txend - cds_end.loc[gene, 'position']]
        else:
            start_stop[gene] = [0, txend - txstart]
    elif canonical.loc[gene, 3] == '+':
        seq = genome[chrom][canonical.loc[gene, 4] - 1:canonical.loc[gene, 5]].seq
        if args.start_cds != 'none' and args.stop_cds != 'none':
            start_stop[gene] = [cds_start.loc[gene, 'position'] - txstart, cds_end.loc[gene, 'position'] - txstart]
        else:
            start_stop[gene] = [0, txend - txstart]
    else:
        print(gene, "strand error")

    genes[gene] = SeqRecord(seq, name=name, id=geneID, description=description)

# Additional parameters
sreEffect = 80
sreEffect3_intron = sreEffect + 19
sreEffect3_exon = sreEffect + 3
sreEffect5_intron = sreEffect + 5
sreEffect5_exon = sreEffect + 3
np.seterr(divide='ignore')

kmer = 6
E = 0
I = 1
B5 = 5
B3 = 3
train_size = 4000
test_size = 1000
score_learning_rate = .01

# Get training, validation, generalization, and test sets
if 'hg19' in args.canonical_ss:
    testGenes = canonical[(canonical[1] == 0) & canonical[2].isin(['chr1', 'chr3', 'chr5', 'chr7', 'chr9'])].index
elif 'TAIR10' in args.canonical_ss:
    testGenes = canonical[(canonical[1] == 0) & canonical[2].isin(['chr2', 'chr4', 'chr1'])].index
elif 'mm10' in args.canonical_ss:
    testGenes = canonical[(canonical[1] == 0) & canonical[2].isin(
        ['chr1', 'chr3', 'chr5', 'chr7', 'chr9', 'chr11', 'chr13', 'chr15', 'chr17'])].index
else:
    testGenes = canonical[(canonical[1] == 0) & canonical[2].isin(
        ['chr1', 'chr3', 'chr5', 'chr7', 'chr9', 'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21', 'chr23'])].index
# testGenes = canonical[(canonical[1] == 0)&canonical[2].isin(['chr2L', 'chr3L'])].index

testGenes = np.intersect1d(testGenes, list(genes.keys()))

trainGenes = canonical.index
trainGenes = np.intersect1d(trainGenes, list(genes.keys()))
trainGenes = np.setdiff1d(trainGenes, testGenes)

lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
trainGenes = trainGenes[lengthsOfGenes > sreEffect3_intron]
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
validationGenes = trainGenes[lengthsOfGenes < 200000]  # < 200000]

generalizationGenes = np.intersect1d(validationGenes, canonical[canonical[1] == 0].index)
if len(generalizationGenes) > test_size: generalizationGenes = np.random.choice(generalizationGenes, test_size,
                                                                                replace=False)
validationGenes = np.setdiff1d(validationGenes, generalizationGenes)
if len(validationGenes) > test_size: validationGenes = np.random.choice(validationGenes, test_size, replace=False)

trainGenes = np.setdiff1d(trainGenes, generalizationGenes)
trainGenes = np.setdiff1d(trainGenes, validationGenes)
if len(trainGenes) > train_size: trainGenes = np.random.choice(trainGenes, train_size, replace=False)
# print(len(trainGenes),len(validationGenes),len(testGenes))

canonical_phases = {}
cannonical_annotations = {}
annotations = {}
for gene in genes.keys():
    phases = []
    annnotation = []
    info = genes[gene].description.split(' ')

    if canonical.loc[gene, 6] == ',':
        cannonical_annotations[gene] = [(0, int(info[5][5:]) - int(info[4][6:]) - 1)]
        continue

    exonEnds = [int(start) - 1 for start in canonical.loc[gene, 6].split(',')[:-1]] + [
        int(info[5][5:])]  # intron starts -> exon ends
    exonStarts = [int(info[4][6:])] + [int(end) + 1 for end in
                                       canonical.loc[gene, 7].split(',')[:-1]]  # exon starts -> intron ends

    exonEnds[-1] -= 1
    exonStarts[0] += 2

    cdslen = 0

    if info[6] == 'Strand:-':
        stop = int(info[5][5:])
        for i in range(len(exonEnds), 0, -1):
            annnotation.append((stop - exonEnds[i - 1] - 1, stop - exonStarts[i - 1] + 1))
            if args.use_periodic_pattern:
                newcds = 0
                if cdslen == 0:
                    if (exonEnds[i - 1] + 1) >= cds_start.loc[gene, 'position'] >= (exonStarts[i - 1] - 1):
                        if cds_end.loc[gene, 'position'] < (exonStarts[i - 1] + 1):
                            newcds = cds_start.loc[gene, 'position'] - exonStarts[i - 1] + 2
                            phases.append(
                                [-1 for j in range(exonEnds[i - 1] - cds_start.loc[gene, 'position'] + 1)] + [j % 3 for
                                                                                                              j in
                                                                                                              range(
                                                                                                                  newcds)])
                        else:
                            newcds = cds_end.loc[gene, 'position'] - cds_start.loc[gene, 'position'] + 1
                            phases.append(
                                [-1 for j in range(exonEnds[i - 1] - cds_start.loc[gene, 'position'] + 1)] + [j % 3 for
                                                                                                              j in
                                                                                                              range(
                                                                                                                  newcds)] + [
                                    -1 for j in range(cds_end.loc[gene, 'position'] - exonStarts[i - 1] + 2)])
                    else:
                        if (exonEnds[i - 1] + 1) < cds_start.loc[gene, 'position'] and (exonStarts[i - 1] - 1) > \
                                cds_end.loc[gene, 'position']:
                            newcds = exonEnds[i - 1] - exonStarts[i - 1] + 3
                            phases.append([(j) % 3 for j in range(newcds)])
                        elif (exonEnds[i - 1] + 1) < cds_start.loc[gene, 'position'] and (exonStarts[i - 1] - 1) <= \
                                cds_end.loc[gene, 'position']:
                            newcds = exonEnds[i - 1] - cds_end.loc[gene, 'position'] + 1
                            phases.append([(cdslen + j) % 3 for j in range(newcds)] + [-1 for j in range(
                                cds_end.loc[gene, 'position'] - exonStarts[i - 1] + 2)])
                    cdslen += newcds
                else:
                    if cds_end.loc[gene, 'position'] < (exonStarts[i - 1] - 1):
                        newcds = (exonEnds[i - 1] - exonStarts[i - 1]) + 3
                        phases.append([(cdslen + j) % 3 for j in range(newcds)])
                    elif (exonStarts[i - 1] - 1) <= cds_end.loc[gene, 'position'] <= (exonEnds[i - 1] + 1):
                        newcds = (exonEnds[i - 1] - cds_end.loc[gene, 'position']) + 1
                        phases.append([(cdslen + j) % 3 for j in range(newcds)] + [-1 for j in range(
                            cds_end.loc[gene, 'position'] - exonStarts[i - 1] + 2)])

                    cdslen += newcds
                if newcds == 0:
                    phases.append([-1 for j in range(exonEnds[i - 1] - exonStarts[i - 1] + 3)])

    elif info[6] == 'Strand:+':
        start = int(info[4][6:])
        for i in range(len(exonEnds)):
            annnotation.append((exonStarts[i] - start - 2, exonEnds[i] - start))
            if args.use_periodic_pattern:
                newcds = 0
                if cdslen == 0:
                    if (exonStarts[i] - 1) <= cds_start.loc[gene, 'position'] <= (exonEnds[i] + 1):
                        if (exonEnds[i] - 1) < cds_end.loc[gene, 'position']:
                            newcds = exonEnds[i] - cds_start.loc[gene, 'position'] + 2
                            phases.append(
                                [-1 for j in range(cds_start.loc[gene, 'position'] - exonStarts[i] + 1)] + [j % 3 for j
                                                                                                            in range(
                                        newcds)])
                        else:
                            newcds = cds_end.loc[gene, 'position'] - cds_start.loc[gene, 'position'] + 1
                            phases.append(
                                [-1 for j in range(cds_start.loc[gene, 'position'] - exonStarts[i] + 1)] + [j % 3 for j
                                                                                                            in range(
                                        newcds)] + [-1 for j in range(exonEnds[i] - cds_end.loc[gene, 'position'] + 2)])
                    else:
                        if (exonStarts[i] - 1) > cds_start.loc[gene, 'position'] and (exonEnds[i] + 1) < cds_end.loc[
                            gene, 'position']:
                            newcds = exonEnds[i] - exonStarts[i] + 3
                            phases.append([(j) % 3 for j in range(newcds)])
                        elif (exonStarts[i] - 1) > cds_start.loc[gene, 'position'] and (exonEnds[i] + 1) >= cds_end.loc[
                            gene, 'position']:
                            newcds = cds_end.loc[gene, 'position'] - exonStarts[i] + 1
                            phases.append([(j) % 3 for j in range(newcds)] + [-1 for j in range(
                                exonEnds[i] - cds_end.loc[gene, 'position'] + 2)])
                    cdslen += newcds
                else:
                    if (exonEnds[i] + 1) < cds_end.loc[gene, 'position']:
                        newcds = exonEnds[i] - exonStarts[i] + 3
                        phases.append([(cdslen + j) % 3 for j in range(newcds)])
                    elif (exonEnds[i] + 1) >= cds_end.loc[gene, 'position'] >= (exonStarts[i] - 1):
                        newcds = cds_end.loc[gene, 'position'] - exonStarts[i] + 1
                        phases.append([(cdslen + j) % 3 for j in range(newcds)] + [-1 for j in range(
                            exonEnds[i] - cds_end.loc[gene, 'position'] + 2)])
                    cdslen += newcds

                if newcds == 0:
                    phases.append([-1 for j in range(exonEnds[i] - exonStarts[i] + 3)])

    cannonical_annotations[gene] = annnotation
    canonical_phases[gene] = phases
    annotations[gene] = {'Canonical': annnotation}

if args.use_periodic_pattern:
    trueSeqs, phaseSeqs = trueSequencesCannonical_C(genes, cannonical_annotations, canonical_phases, E, I, B3, B5)
else:
    trueSeqs = trueSequencesCannonical(genes, cannonical_annotations, E, I, B3, B5)

# Structural parameters
numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons \
    = structuralParameters(trainGenes, annotations)

N = int(np.ceil(max(numExonsPerGene) / 10.0) * 10)
numExonsHist = np.histogram(numExonsPerGene, bins=N, range=(0, N))[0]  # nth bin is [n-1,n) bun Nth bin is [N-1,N]
numExonsDF = pd.DataFrame(list(zip(np.arange(0, N), numExonsHist)), columns=['Index', 'Count'])

p1E = float(numExonsDF['Count'][1]) / numExonsDF['Count'].sum()
pEO = float(numExonsDF['Count'][2:].sum()) / (numExonsDF['Index'][2:] * numExonsDF['Count'][2:]).sum()
numExonsDF['Prob'] = [pEO * (1 - pEO) ** (i - 2) for i in range(0, N)]
numExonsDF.loc[1, 'Prob'] = p1E
numExonsDF.loc[0, 'Prob'] = 0
numExonsDF['Empirical'] = numExonsDF['Count'] / numExonsDF['Count'].sum()
transitions = [1 - p1E, 1 - pEO]

N = 5000000
lengthIntronsHist = np.histogram(lengthIntrons, bins=N, range=(0, N))[0]  # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthIntronsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthIntronsHist)), columns=['Index', 'Count'])
lengthIntronsDF['Prob'] = lengthIntronsDF['Count'] / lengthIntronsDF['Count'].sum()
if 'hg19' in args.canonical_ss or 'mm10' in args.canonical_ss:
    pIL = geometric_smooth_tailed(lengthIntrons, N, 15, 1000, lower_cutoff=60)
elif 'TAIR10' in args.canonical_ss:
    pIL = geometric_smooth_tailed(lengthIntrons, N, 5, 200, lower_cutoff=60)
else:
    pIL = geometric_smooth_tailed(lengthIntrons, N, 5, 5000, lower_cutoff=50)

lengthFirstExonsHist = np.histogram(lengthFirstExons, bins=N, range=(0, N))[
    0]  # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthFirstExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthFirstExonsHist)), columns=['Index', 'Count'])
lengthFirstExonsDF['Prob'] = lengthFirstExonsDF['Count'] / lengthFirstExonsDF['Count'].sum()
pELF = adaptive_kde_tailed(lengthFirstExons, N)

lengthMiddleExonsHist = np.histogram(lengthMiddleExons, bins=N, range=(0, N))[
    0]  # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthMiddleExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthMiddleExonsHist)), columns=['Index', 'Count'])
lengthMiddleExonsDF['Prob'] = lengthMiddleExonsDF['Count'] / lengthMiddleExonsDF['Count'].sum()
pELM = adaptive_kde_tailed(lengthMiddleExons, N)

lengthLastExonsHist = np.histogram(lengthLastExons, bins=N, range=(0, N))[
    0]  # nth bin is [n-1,n) bun Nth bin is [N-1,N]
lengthLastExonsDF = pd.DataFrame(list(zip(np.arange(0, N), lengthLastExonsHist)), columns=['Index', 'Count'])
lengthLastExonsDF['Prob'] = lengthLastExonsDF['Count'] / lengthLastExonsDF['Count'].sum()
pELL = adaptive_kde_tailed(lengthLastExons, N)
pELS = 0 * np.copy(pELL)

if args.use_periodic_pattern:
    sreScores_exon = np.ones((3, 4 ** kmer))
else:
    sreScores_exon = np.ones(4 ** kmer)
sreScores_intron = np.ones(4 ** kmer)

if args.prelearned_sres != 'none':
    sreScores_exon = np.load(os.path.join(args.prelearned_sres, "sreScores_exon.npy"))
    sreScores3_exon = np.load(os.path.join(args.prelearned_sres, "sreScores3_exon.npy"))
    sreScores5_exon = np.load(os.path.join(args.prelearned_sres, "sreScores5_exon.npy"))
    sreScores_intron = np.load(os.path.join(args.prelearned_sres, "sreScores_intron.npy"))
    sreScores3_intron = np.load(os.path.join(args.prelearned_sres, "sreScores3_intron.npy"))
    sreScores5_intron = np.load(os.path.join(args.prelearned_sres, "sreScores5_intron.npy"))
    rfScores_open = 1.0
    if args.use_reading_frame:
        rfScores_close = 1.0 * 10 ** 9
    else:
        rfScores_close = 1.0
    if args.use_periodic_pattern:
        phi = np.array([0.4, 0.3, 0.3])
        phi5 = np.array([0.3, 0.4, 0.3])
    # sreScores_exon = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_exon.npy')
    # sreScores_intron = np.load('organism_parameters/' + args.prelearned_sres + '/sreScores_intron.npy')
    # rfScores_open = np.load('organism_parameters/' + args.prelearned_sres + '/rfScores_open.npy')
    # rfScores_open = np.load('organism_parameters/' + args.prelearned_sres + '/rfScores_close.npy')
else:
    sreScores3_exon = np.copy(sreScores_exon)
    sreScores5_exon = np.copy(sreScores_exon)
    sreScores3_intron = np.copy(sreScores_intron)
    sreScores5_intron = np.copy(sreScores_intron)
    # rfScores_open = np.copy(rfScores_open)
    # rfScores_close = np.copy(rfScores_close)

# Learning seed
if args.learning_seed == 'real-decoy' and args.learn_sres:
    me5 = maxEnt5(trainGenes, genes, maxEntDir)
    me3 = maxEnt3(trainGenes, genes, maxEntDir)

    tolerance = .5
    decoySS = {}
    for gene in trainGenes:
        decoySS[gene] = np.zeros(len(genes[gene]), dtype=int)

    # 5'SS
    five_scores = []
    for gene in trainGenes:
        for score in np.log2(me5[gene][trueSeqs[gene] == B5][1:]): five_scores.append(score)
    five_scores = np.array(five_scores)

    five_scores_tracker = np.flip(np.sort(list(five_scores)))

    for score in five_scores_tracker:
        np.random.shuffle(trainGenes)
        g = 0
        while g < len(trainGenes):
            gene = trainGenes[g]
            g += 1
            true_ss = get_all_5ss(gene, allSS, genes)
            used_sites = np.nonzero(decoySS[gene] == B5)[0]

            gene5s = np.log2(me5[gene])
            sort_inds = np.argsort(gene5s)
            sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
            sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
            L = len(sort_inds)
            gene5s = gene5s[sort_inds]

            up_i = np.searchsorted(gene5s, score, 'left')
            down_i = up_i - 1
            if down_i >= L: down_i = L - 1
            if up_i >= L: up_i = L - 1

            if abs(score - gene5s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                decoySS[gene][sort_inds[down_i]] = B5
                g = len(trainGenes)

            elif abs(score - gene5s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                decoySS[gene][sort_inds[up_i]] = B5
                g = len(trainGenes)

    # 3'SS
    three_scores = []
    for gene in trainGenes:
        for score in np.log2(me3[gene][trueSeqs[gene] == B3][:-1]): three_scores.append(score)
    three_scores = np.array(three_scores)

    three_scores_tracker = np.flip(np.sort(list(three_scores)))

    for score in three_scores_tracker:
        np.random.shuffle(trainGenes)
        g = 0
        while g < len(trainGenes):
            gene = trainGenes[g]
            g += 1
            true_ss = get_all_3ss(gene, allSS, genes)
            used_sites = np.nonzero(decoySS[gene] == B3)[0]

            gene3s = np.log2(me3[gene])
            sort_inds = np.argsort(gene3s)
            sort_inds = sort_inds[~np.in1d(sort_inds, true_ss)]
            sort_inds = sort_inds[~np.in1d(sort_inds, used_sites)]
            L = len(sort_inds)
            gene3s = gene3s[sort_inds]

            up_i = np.searchsorted(gene3s, score, 'left')
            down_i = up_i - 1
            if down_i >= L: down_i = L - 1
            if up_i >= L: up_i = L - 1

            if abs(score - gene3s[down_i]) < tolerance and decoySS[gene][sort_inds[down_i]] == 0:
                decoySS[gene][sort_inds[down_i]] = B3
                g = len(trainGenes)

            elif abs(score - gene3s[up_i]) < tolerance and decoySS[gene][sort_inds[up_i]] == 0:
                decoySS[gene][sort_inds[up_i]] = B3
                g = len(trainGenes)
    if args.use_periodic_pattern:
        (sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon, phi,
         phi5) = get_hexamer_real_decoy_scores_C(trainGenes, trueSeqs, decoySS, genes, phaseSeqs, kmer=kmer,
                                                 sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                                                 sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                                                 start_stop=start_stop)
    else:
        (sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon,
         rfScores_open, rfScores_close) = get_hexamer_real_decoy_scores(trainGenes, trueSeqs, decoySS, genes, kmer=kmer,
                                                                        sreEffect5_exon=sreEffect5_exon,
                                                                        sreEffect5_intron=sreEffect5_intron,
                                                                        sreEffect3_exon=sreEffect3_exon,
                                                                        sreEffect3_intron=sreEffect3_intron,
                                                                        start_stop=start_stop)
    if args.split_intron == False:
        sreScores3_intron = sreScores_intron
        sreScores5_intron = np.copy(sreScores_intron)
    # sreScores3_intron = sreScores_intron
    # sreScores3_exon = sreScores_exon
    # sreScores5_intron = np.copy(sreScores_intron)
    # sreScores5_exon = np.copy(sreScores_exon)
    '''if args.use_periodic_pattern!='True':
        sreScores3_exon = sreScores_exon
        sreScores5_exon = np.copy(sreScores_exon)'''
    if args.split_exon == False:
        sreScores3_exon = sreScores_exon
        sreScores5_exon = np.copy(sreScores_exon)
    #######################################################
    rfScores_open = 1.0
    if args.use_reading_frame:
        rfScores_close = 1.0 * 10 ** 9
    else:
        rfScores_close = 1.0
    #######################################################

    with open(os.path.join(pickle_path, "initial_scores.pickle"), "wb") as handle:
        pickle.dump({'sreScore_intron': sreScores_intron, 'sreScores3_intron': sreScores3_intron,
                     'sreScores5_intron': sreScores5_intron, 'sreScores_exon': sreScores_exon,
                     'sreScores3_exon': sreScores3_exon, 'sreScores5_exon': sreScores5_exon,
                     'rfScore_open': rfScores_open, 'rfScore_close': rfScores_close}, handle)
    # Learn weight

    if args.threads > 0:
        validationGenes = order_genes(validationGenes, args.threads, genes)
    lengths = np.array([len(str(genes[gene].seq)) for gene in validationGenes])
    sequences = [str(genes[gene].seq) for gene in validationGenes]
    coordinates_fl = np.array(
        [[np.nonzero(trueSeqs[gene] == 5)[0][0], np.nonzero(trueSeqs[gene] == 3)[0][-1]] for gene in
         validationGenes])
    start_stop_gene = np.array([start_stop[gene] for gene in validationGenes])

    step_size = 1
    sre_weights = [0, step_size]
    scores = []
    for sre_weight in sre_weights:
        if args.use_periodic_pattern:
            exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
            exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
        else:
            exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
        intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
        intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

        # rfScoreOpen = np.zeros(len(sequences))
        # rfScoreClose = np.zeros(len(sequences))

        for g, sequence in enumerate(sequences):
            if args.use_periodic_pattern:
                exonic5s_all = np.array(
                    sreScores_single_phase(sequence.lower(), np.exp(np.log(sreScores5_exon) * sre_weight), kmer))
                exonic3s_all = np.array(
                    sreScores_single_phase(sequence.lower(), np.exp(np.log(sreScores3_exon) * sre_weight), kmer))
                for phs in [0, 1, 2]:
                    exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
                    exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
            else:
                exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_exon) * sre_weight), kmer)))
                exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_exon) * sre_weight), kmer)))
            intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_intron) * sre_weight), kmer)))
            intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_intron) * sre_weight), kmer)))
        # rfScores_open = np.exp(np.log(rfScores_open)*sre_weight)
        # rfScores_close = np.exp(np.log(rfScores_close)*sre_weight)

        if args.use_periodic_pattern:
            pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                                 pELS=pELS, pELF=pELF, pELM=pELM,
                                 pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                                 intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                                 sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                                 sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, rfClose=1.0,
                                 start_stop=start_stop_gene, meDir=maxEntDir)
        else:
            pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                               pELS=pELS, pELF=pELF, pELM=pELM,
                               pELL=pELL, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                               intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                               sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                               sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, rfOpen=1.0,
                               rfClose=1.0, start_stop=start_stop_gene, meDir=maxEntDir)

        # Get the Sensitivity and Precision
        num_truePositives = 0
        num_falsePositives = 0
        num_falseNegatives = 0

        for g, gene in enumerate(validationGenes):
            L = lengths[g]
            predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
            trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

            predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
            trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

            trueThree1 = trueThrees[-1]
            trueThrees = trueThrees[:-1]

            trueFive1 = trueFives[0]
            trueFives = trueFives[1:]

            predThrees = predThrees[:-1]
            predFives = predFives[0:]

            ind_true_3 = np.where(trueThrees > start_stop[gene][0])
            ind_true_5 = np.where(trueFives < start_stop[gene][1])
            # ind_true=np.intersect1d(ind_true_3,ind_true_5)

            trueThrees = trueThrees[ind_true_3]
            trueFives = trueFives[ind_true_5]

            ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
            ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
            # int_pred=np.intersect1d(ind_pred_3,ind_pred_5)
            predThrees = predThrees[ind_pred_3]
            predFives = predFives[ind_pred_5]

            num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))
            num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(np.setdiff1d(predFives, trueFives))
            num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(np.setdiff1d(trueFives, predFives))

        ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
        ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
        f1 = 2 / (1 / ssSens + 1 / ssPrec)
        scores.append(f1)

    scores = np.array(scores)
    sre_weights = np.array(sre_weights)

    while len(scores) < 15:
        i = np.argmax(scores)
        if i == len(scores) - 1:
            sre_weights_test = [sre_weights[-1] + step_size]
        elif i == 0:
            sre_weights_test = [sre_weights[1] / 2]
        else:
            sre_weights_test = [sre_weights[i] / 2 + sre_weights[i - 1] / 2,
                                sre_weights[i] / 2 + sre_weights[i + 1] / 2]

        for sre_weight in sre_weights_test:
            sre_weights = np.append(sre_weights, sre_weight)
            if args.use_periodic_pattern:
                exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
            else:
                exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

            for g, sequence in enumerate(sequences):
                if args.use_periodic_pattern:
                    exonic5s_all = np.array(
                        sreScores_single_phase(sequence.lower(), np.exp(np.log(sreScores5_exon) * sre_weight), kmer))
                    exonic3s_all = np.array(
                        sreScores_single_phase(sequence.lower(), np.exp(np.log(sreScores3_exon) * sre_weight), kmer))
                    for phs in [0, 1, 2]:
                        exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
                        exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
                else:
                    exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(
                            sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_exon) * sre_weight), kmer)))
                    exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(
                            sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_exon) * sre_weight), kmer)))
                intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores5_intron) * sre_weight), kmer)))
                intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), np.exp(np.log(sreScores3_intron) * sre_weight), kmer)))
            # rfScores_open = np.exp(np.log(rfScores_open)*sre_weight)
            # rfScores_close = np.exp(np.log(rfScores_close)*sre_weight)

            if args.use_periodic_pattern:
                pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl,
                                     pIL=pIL, pELS=pELS, pELF=pELF,
                                     pELM=pELM, pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s,
                                     exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s,
                                     intronicSREs3s=intronicSREs3s, k=kmer, sreEffect5_exon=sreEffect5_exon,
                                     sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon,
                                     sreEffect3_intron=sreEffect3_intron, start_stop=start_stop_gene, meDir=maxEntDir)
            else:
                pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                                   pELS=pELS, pELF=pELF, pELM=pELM,
                                   pELL=pELL, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                                   intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                                   sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                                   sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, rfOpen=1.0,
                                   rfClose=1.0, start_stop=start_stop_gene, meDir=maxEntDir)

            # Get the Sensitivity and Precision
            num_truePositives = 0
            num_falsePositives = 0
            num_falseNegatives = 0

            for g, gene in enumerate(validationGenes):
                L = lengths[g]
                predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
                trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

                predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
                trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

                trueThree1 = trueThrees[-1]
                trueThrees = trueThrees[:-1]

                trueFive1 = trueFives[0]
                trueFives = trueFives[1:]

                predThrees = predThrees[:-1]
                predFives = predFives[0:]

                ind_true_3 = np.where(trueThrees > start_stop[gene][0])
                ind_true_5 = np.where(trueFives < start_stop[gene][1])
                # ind_true = np.intersect1d(ind_true_3, ind_true_5)

                trueThrees = trueThrees[ind_true_3]
                trueFives = trueFives[ind_true_5]

                ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
                ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
                # int_pred = np.intersect1d(ind_pred_3, ind_pred_5)
                predThrees = predThrees[ind_pred_3]
                predFives = predFives[ind_pred_5]

                num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(
                    np.intersect1d(predFives, trueFives))
                num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(
                    np.setdiff1d(predFives, trueFives))
                num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(
                    np.setdiff1d(trueFives, predFives))

            ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
            ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
            f1 = 2 / (1 / ssSens + 1 / ssPrec)
            scores = np.append(scores, f1)

        scores = scores[np.argsort(sre_weights)]
        sre_weights = sre_weights[np.argsort(sre_weights)]

    # Set up scores for score learning
    sre_weight = sre_weights[np.argmax(scores)]

    sreScores_exon = np.exp(np.log(sreScores_exon) * sre_weight)
    sreScores5_exon = np.exp(np.log(sreScores5_exon) * sre_weight)
    sreScores3_exon = np.exp(np.log(sreScores3_exon) * sre_weight)
    '''if args.use_periodic_pattern!=True:
        sreScores5_exon=np.copy(sreScores_exon)
        sreScores3_exon=np.copy(sreScores_exon)'''
    sreScores_intron = np.exp(np.log(sreScores_intron) * sre_weight)
    sreScores5_intron = np.exp(np.log(sreScores5_intron) * sre_weight)
    sreScores3_intron = np.exp(np.log(sreScores3_intron) * sre_weight)

    with open(os.path.join(pickle_path, "initial_multiweight_scores.pickle"), "wb") as handle:
        pickle.dump({'sreScore_intron': sreScores_intron, 'sreScore3_intron': sreScores3_intron,
                     'sreScore5_intron': sreScores5_intron, 'sreScores_exon': sreScores_exon,
                     'sreScores3_exon': sreScores3_exon, 'sreScores5_exon': sreScores5_exon,
                     'rfScore_open': rfScores_open, 'rfScore_close': rfScores_close}, handle)

openscores = []
closescores = []
# Learning

if args.learn_sres:
    # Update close only
    samef1 = {}
    update_step = 0
    score_learning_rate_rf = 0.2 / (update_step + 1)
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in trainGenes])
    trainGenesShort = trainGenes[lengthsOfGenes < 200000]  # trainGenes[lengthsOfGenes < 200000]
    np.random.shuffle(trainGenesShort)
    trainGenesShort = np.array_split(trainGenesShort, 4)

    trainGenes1 = trainGenesShort[0]
    trainGenes2 = trainGenesShort[1]
    trainGenes3 = trainGenesShort[2]
    trainGenes4 = trainGenesShort[3]
    trainGenesTest = generalizationGenes
    held_f1 = -1

    if args.use_reading_frame:
        held_rfScores_open = np.copy(rfScores_open)
        held_rfScores_close = np.copy(rfScores_close)
        learning_counter = -1
        doneTime = 10
        while 1 < doneTime:
            learning_counter += 1
            if learning_counter > args.max_learned_scores: break
            update_scores = True

            if learning_counter % 5 == 1:
                trainGenesSub = np.copy(trainGenes1)
            elif learning_counter % 5 == 2:
                trainGenesSub = np.copy(trainGenes2)
            elif learning_counter % 5 == 3:
                trainGenesSub = np.copy(trainGenes3)
            elif learning_counter % 5 == 4:
                trainGenesSub = np.copy(trainGenes4)
            else:
                trainGenesSub = np.copy(trainGenesTest)
                update_scores = False
            if args.threads > 0:
                trainGenesSub = order_genes(trainGenesSub, args.threads, genes)
            lengths = np.array([len(str(genes[gene].seq)) for gene in trainGenesSub])
            sequences = [str(genes[gene].seq) for gene in trainGenesSub]
            coordinates_fl = np.array(
                [[np.nonzero(trueSeqs[gene] == 5)[0][0], np.nonzero(trueSeqs[gene] == 3)[0][-1]] for gene in
                 trainGenesSub])
            start_stop_gene = np.array([start_stop[gene] for gene in trainGenesSub])

            if args.use_periodic_pattern:
                exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
            else:
                exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

            for g, sequence in enumerate(sequences):
                if args.use_periodic_pattern:
                    exonic5s_all = np.array(
                        sreScores_single_phase(sequence.lower(), sreScores5_exon,
                                               kmer))
                    exonic3s_all = np.array(
                        sreScores_single_phase(sequence.lower(), sreScores3_exon,
                                               kmer))
                    for phs in [0, 1, 2]:
                        exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
                        exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
                else:
                    exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
                    exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
                intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
                intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

            if args.use_periodic_pattern:
                pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl,
                                     pIL=pIL, pELS=pELS, pELF=pELF,
                                     pELM=pELM, pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s,
                                     exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s,
                                     intronicSREs3s=intronicSREs3s, k=kmer, sreEffect5_exon=sreEffect5_exon,
                                     sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon,
                                     sreEffect3_intron=sreEffect3_intron, start_stop=start_stop_gene, meDir=maxEntDir,
                                     rfClose=rfScores_close)
            else:
                pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                                   pELS=pELS, pELF=pELF, pELM=pELM, pELL=pELL, exonicSREs5s=exonicSREs5s,
                                   exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s,
                                   intronicSREs3s=intronicSREs3s, k=kmer, sreEffect5_exon=sreEffect5_exon,
                                   sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon,
                                   sreEffect3_intron=sreEffect3_intron, rfOpen=rfScores_open, rfClose=rfScores_close,
                                   start_stop=start_stop_gene, meDir=maxEntDir,
                                   lmode=update_scores)  # does rf open/close need effect value? #alter viterbi!
            # print('pred_all_finish')
            # Get the False Negatives and False Positives
            falsePositives = {}
            falseNegatives = {}

            num_truePositives = 0
            num_falsePositives = 0
            num_falseNegatives = 0

            for g, gene in enumerate(trainGenesSub):
                L = lengths[g]
                falsePositives[gene] = np.zeros(len(genes[gene]), dtype=int)
                falseNegatives[gene] = np.zeros(len(genes[gene]), dtype=int)

                predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
                trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

                predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
                trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

                trueThree1 = trueThrees[-1]
                trueThrees = trueThrees[:-1]

                trueFive1 = trueFives[0]
                trueFives = trueFives[1:]

                predThrees = predThrees[:-1]
                predFives = predFives[0:]

                ind_true_3 = np.where(trueThrees > start_stop[gene][0])
                ind_true_5 = np.where(trueFives < start_stop[gene][1])
                # ind_true = np.intersect1d(ind_true_3, ind_true_5)

                trueThrees = trueThrees[ind_true_3]
                trueFives = trueFives[ind_true_5]

                ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
                ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
                # int_pred = np.intersect1d(ind_pred_3, ind_pred_5)
                predThrees = predThrees[ind_pred_3]
                predFives = predFives[ind_pred_5]

                num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(
                    np.intersect1d(predFives, trueFives))

                falsePositives[gene][np.setdiff1d(predThrees, trueThrees)] = B3
                falsePositives[gene][np.setdiff1d(predFives, trueFives)] = B5
                num_falsePositives += np.sum(falsePositives[gene] > 0)

                falseNegatives[gene][np.setdiff1d(trueThrees, predThrees)] = B3
                falseNegatives[gene][np.setdiff1d(trueFives, predFives)] = B5
                num_falseNegatives += np.sum(falseNegatives[gene] > 0)

            ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
            ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
            f1 = 2 / (1 / ssSens + 1 / ssPrec)
            if update_scores:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close = get_hexamer_counts(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)

                set2_counts_open += 1
                set1_counts_open += 1

                set2_counts_close += 1
                set1_counts_close += 1

                psuedocount_denominator_open = set1_counts_open + set2_counts_open
                set1_counts_open = set1_counts_open + set1_counts_open / psuedocount_denominator_open
                set2_counts_open = set2_counts_open + set2_counts_open / psuedocount_denominator_open

                frequency_ratio_open = set1_counts_open / set2_counts_open

                psuedocount_denominator_close = set1_counts_close + set2_counts_close
                set1_counts_close = set1_counts_close / (
                            set1_counts_close + set1_counts_open)  # set1_counts_close / psuedocount_denominator_close
                set2_counts_close = set2_counts_close / (
                            set2_counts_close + set2_counts_open)  # set2_counts_close / psuedocount_denominator_close

                frequency_ratio_close = 1 / (set1_counts_close / set2_counts_close)
                rfScores_close *= frequency_ratio_close ** (score_learning_rate_rf)


            else:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close = get_hexamer_counts(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon,
                    sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)

                if f1 > held_f1 or (f1 == held_f1 and (
                        set1_counts_close != 0 or set2_counts_close != 0)):  # and samef1[f1]>5):  # Hold onto the scores with the highest f1 performance
                    held_f1 = np.copy(f1)
                    '''held_sreScores5_exon = np.copy(sreScores5_exon)
                    held_sreScores3_exon = np.copy(sreScores3_exon)
                    held_sreScores5_intron = np.copy(sreScores5_intron)
                    held_sreScores3_intron = np.copy(sreScores3_intron)'''
                    held_rfScores_open = np.copy(rfScores_open)
                    held_rfScores_close = np.copy(rfScores_close)
                    if f1 not in samef1:
                        samef1[f1] = 1
                    else:
                        samef1[f1] += 1
                    if samef1[f1] == 5:
                        doneTime = 0
                else:
                    doneTime = 0  # Stop learning if the performance has decreased

                '''sreScores5_exon = np.copy(held_sreScores5_exon)
                sreScores3_exon = np.copy(held_sreScores3_exon)
                sreScores5_intron = np.copy(held_sreScores5_intron)
                sreScores3_intron = np.copy(held_sreScores3_intron)'''
                rfScores_open = np.copy(held_rfScores_open)
                rfScores_close = np.copy(held_rfScores_close)

            with open(os.path.join(pickle_path, "updated_multiweight_scores_step1_" + str(update_step) + ".pickle"),
                      "wb") as handle:
                pickle.dump({'sreScore_intron': sreScores_intron, 'sreScores3_intron': sreScores3_intron,
                             'sreScores5_intron': sreScores5_intron, 'sreScores_exon': sreScores_exon,
                             'sreScores3_exon': sreScores3_exon, 'sreScores5_exon': sreScores5_exon,
                             'rfScore_open': rfScores_open, 'rfScore_close': rfScores_close}, handle)
            update_step += 1
            performance.append([1, update_step, ssSens, ssPrec, f1])

    # Update SRE only
    update_step = 0
    held_sreScores5_exon = np.copy(sreScores5_exon)
    held_sreScores3_exon = np.copy(sreScores3_exon)
    held_sreScores5_intron = np.copy(sreScores5_intron)
    held_sreScores3_intron = np.copy(sreScores3_intron)
    if args.use_reading_frame:
        learning_counter = 0
    else:
        learning_counter = -1
    doneTime = 10
    while 1 < doneTime:
        learning_counter += 1
        if learning_counter > args.max_learned_scores: break
        update_scores = True

        if learning_counter % 5 == 1:
            trainGenesSub = np.copy(trainGenes1)
        elif learning_counter % 5 == 2:
            trainGenesSub = np.copy(trainGenes2)
        elif learning_counter % 5 == 3:
            trainGenesSub = np.copy(trainGenes3)
        elif learning_counter % 5 == 4:
            trainGenesSub = np.copy(trainGenes4)
        else:
            trainGenesSub = np.copy(trainGenesTest)
            update_scores = False
        if args.threads > 0:
            trainGenesSub = order_genes(trainGenesSub, args.threads, genes)
        lengths = np.array([len(str(genes[gene].seq)) for gene in trainGenesSub])
        sequences = [str(genes[gene].seq) for gene in trainGenesSub]
        coordinates_fl = np.array(
            [[np.nonzero(trueSeqs[gene] == 5)[0][0], np.nonzero(trueSeqs[gene] == 3)[0][-1]] for gene in
             trainGenesSub])
        start_stop_gene = np.array([start_stop[gene] for gene in trainGenesSub])

        if args.use_periodic_pattern:
            exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
            exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
        else:
            exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
        intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
        intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

        for g, sequence in enumerate(sequences):
            if args.use_periodic_pattern:
                exonic5s_all = np.array(sreScores_single_phase(sequence.lower(), sreScores5_exon, kmer))
                exonic3s_all = np.array(sreScores_single_phase(sequence.lower(), sreScores3_exon, kmer))
                for phs in [0, 1, 2]:
                    exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
                    exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
            else:
                exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
                exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
            intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
            intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

        if args.use_reading_frame:
            rfScores_close = 1.0
        if args.use_periodic_pattern:
            pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl,
                                 pIL=pIL, pELS=pELS, pELF=pELF,
                                 pELM=pELM, pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s,
                                 exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s,
                                 intronicSREs3s=intronicSREs3s, k=kmer, sreEffect5_exon=sreEffect5_exon,
                                 sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon,
                                 sreEffect3_intron=sreEffect3_intron, start_stop=start_stop_gene,
                                 meDir=maxEntDir, rfClose=rfScores_close)
        else:
            pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                               pELS=pELS, pELF=pELF, pELM=pELM,
                               pELL=pELL, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                               intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                               sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                               sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                               rfOpen=rfScores_open,
                               rfClose=rfScores_close, start_stop=start_stop_gene, meDir=maxEntDir,
                               lmode=update_scores)  # does rf open/close need effect value? #alter viterbi!
        # Get the False Negatives and False Positives
        falsePositives = {}
        falseNegatives = {}

        num_truePositives = 0
        num_falsePositives = 0
        num_falseNegatives = 0

        for g, gene in enumerate(trainGenesSub):
            L = lengths[g]
            falsePositives[gene] = np.zeros(len(genes[gene]), dtype=int)
            falseNegatives[gene] = np.zeros(len(genes[gene]), dtype=int)

            predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
            trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

            predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
            trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

            trueThree1 = trueThrees[-1]
            trueThrees = trueThrees[:-1]

            trueFive1 = trueFives[0]
            trueFives = trueFives[1:]

            predThrees = predThrees[:-1]
            predFives = predFives[0:]

            ind_true_3 = np.where(trueThrees > start_stop[gene][0])
            ind_true_5 = np.where(trueFives < start_stop[gene][1])
            # ind_true = np.intersect1d(ind_true_3, ind_true_5)

            trueThrees = trueThrees[ind_true_3]
            trueFives = trueFives[ind_true_5]

            ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
            ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
            # int_pred = np.intersect1d(ind_pred_3, ind_pred_5)
            predThrees = predThrees[ind_pred_3]
            predFives = predFives[ind_pred_5]

            num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))

            falsePositives[gene][np.setdiff1d(predThrees, trueThrees)] = B3
            falsePositives[gene][np.setdiff1d(predFives, trueFives)] = B5
            num_falsePositives += np.sum(falsePositives[gene] > 0)

            falseNegatives[gene][np.setdiff1d(trueThrees, predThrees)] = B3
            falseNegatives[gene][np.setdiff1d(trueFives, predFives)] = B5
            num_falseNegatives += np.sum(falseNegatives[gene] > 0)

        ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
        ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
        f1 = 2 / (1 / ssSens + 1 / ssPrec)
        if update_scores:
            if args.use_periodic_pattern:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, phicount, fpcount = get_hexamer_counts_C(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, phaseSeqs, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)
                set1_counts_intron = set1_counts_5_intron + set1_counts_3_intron
                set2_counts_intron = set2_counts_5_intron + set2_counts_3_intron

                psuedocount_denominator_intron = np.sum(set1_counts_intron) + np.sum(set2_counts_intron)
                set1_counts_intron = set1_counts_intron + np.sum(set1_counts_intron) / psuedocount_denominator_intron
                set2_counts_intron = set2_counts_intron + np.sum(set2_counts_intron) / psuedocount_denominator_intron

                frequency_ratio_intron = set1_counts_intron / np.sum(set1_counts_intron) / (
                        set2_counts_intron / np.sum(set2_counts_intron))

                sreScores_intron *= frequency_ratio_intron ** score_learning_rate

                # separately
                psuedocount_denominator_intron5 = np.sum(set1_counts_5_intron) + np.sum(set2_counts_5_intron)
                set1_counts_5_intron = set1_counts_5_intron + np.sum(
                    set1_counts_5_intron) / psuedocount_denominator_intron5
                set2_counts_5_intron = set2_counts_5_intron + np.sum(
                    set2_counts_5_intron) / psuedocount_denominator_intron5

                frequency_ratio_intron5 = set1_counts_5_intron / np.sum(set1_counts_5_intron) / (
                        set2_counts_5_intron / np.sum(set2_counts_5_intron))
                sreScores5_intron *= frequency_ratio_intron5 ** score_learning_rate

                psuedocount_denominator_intron3 = np.sum(set1_counts_3_intron) + np.sum(set2_counts_3_intron)
                set1_counts_3_intron = set1_counts_3_intron + np.sum(
                    set1_counts_3_intron) / psuedocount_denominator_intron3
                set2_counts_3_intron = set2_counts_3_intron + np.sum(
                    set2_counts_3_intron) / psuedocount_denominator_intron3

                frequency_ratio_intron3 = set1_counts_3_intron / np.sum(set1_counts_3_intron) / (
                        set2_counts_3_intron / np.sum(set2_counts_3_intron))
                sreScores3_intron *= frequency_ratio_intron3 ** score_learning_rate

                for phs in [0, 1, 2]:
                    # 5
                    psuedocount_denominator_exon5 = np.sum(set1_counts_5_exon[phs]) + np.sum(set2_counts_5_exon)
                    set1_counts_exon5 = set1_counts_5_exon[phs] + np.sum(
                        set1_counts_5_exon[phs]) / psuedocount_denominator_exon5
                    set2_counts_exon5 = set2_counts_5_exon + np.sum(set2_counts_5_exon) / psuedocount_denominator_exon5

                    frequency_ratio_exon5 = set1_counts_exon5 / np.sum(set1_counts_exon5) / (
                            set2_counts_exon5 / np.sum(set2_counts_exon5))
                    sreScores5_exon[phs] *= frequency_ratio_exon5 ** score_learning_rate

                    # 3
                    psuedocount_denominator_exon3 = np.sum(set1_counts_3_exon[phs]) + np.sum(set2_counts_3_exon)
                    set1_counts_exon3 = set1_counts_3_exon[phs] + np.sum(
                        set1_counts_3_exon[phs]) / psuedocount_denominator_exon3
                    set2_counts_exon3 = set2_counts_3_exon + np.sum(set2_counts_3_exon) / psuedocount_denominator_exon3
                    frequency_ratio_exon3 = set1_counts_exon3 / np.sum(set1_counts_exon3) / (
                            set2_counts_exon3 / np.sum(set2_counts_exon3))
                    sreScores3_exon[phs] *= frequency_ratio_exon3 ** score_learning_rate

                    # 5+3
                    set1_counts_exon_ = set1_counts_5_exon + set1_counts_3_exon
                    set2_counts_exon_ = set2_counts_5_exon + set2_counts_3_exon
                    pseudocount_denominator_exon = np.sum(set1_counts_exon_[phs]) + np.sum(set2_counts_exon_)
                    set1_counts_exon_ = set1_counts_exon_[phs] + np.sum(
                        set1_counts_exon_[phs]) / pseudocount_denominator_exon
                    set2_counts_exon_ = set2_counts_exon_ + np.sum(set2_counts_exon_) / pseudocount_denominator_exon
                    frequency_ratio_exon = set1_counts_exon_ / np.sum(set1_counts_exon_) / (
                            set2_counts_exon_ / np.sum(set2_counts_exon_))
                    sreScores_exon[phs] *= frequency_ratio_exon ** score_learning_rate

                # sreScores3_intron = sreScores_intron
                # sreScores5_intron = sreScores_intron
                if args.split_intron == False:
                    sreScores3_intron = sreScores_intron
                    sreScores5_intron = np.copy(sreScores_intron)
                if args.split_exon == False:
                    sreScores3_exon = sreScores_exon
                    sreScores5_exon = np.copy(sreScores_exon)
            else:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close = get_hexamer_counts(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon,
                    sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)

                set1_counts_intron = set1_counts_5_intron + set1_counts_3_intron
                set1_counts_exon = set1_counts_5_exon + set1_counts_3_exon
                set2_counts_intron = set2_counts_5_intron + set2_counts_3_intron
                set2_counts_exon = set2_counts_5_exon + set2_counts_3_exon

                psuedocount_denominator_intron = np.sum(set1_counts_intron) + np.sum(set2_counts_intron)
                set1_counts_intron = set1_counts_intron + np.sum(set1_counts_intron) / psuedocount_denominator_intron
                set2_counts_intron = set2_counts_intron + np.sum(set2_counts_intron) / psuedocount_denominator_intron

                frequency_ratio_intron = set1_counts_intron / np.sum(set1_counts_intron) / (
                        set2_counts_intron / np.sum(set2_counts_intron))
                sreScores_intron *= frequency_ratio_intron ** score_learning_rate

                # separately
                psuedocount_denominator_intron5 = np.sum(set1_counts_5_intron) + np.sum(set2_counts_5_intron)
                set1_counts_5_intron = set1_counts_5_intron + np.sum(
                    set1_counts_5_intron) / psuedocount_denominator_intron5
                set2_counts_5_intron = set2_counts_5_intron + np.sum(
                    set2_counts_5_intron) / psuedocount_denominator_intron5

                frequency_ratio_intron5 = set1_counts_5_intron / np.sum(set1_counts_5_intron) / (
                        set2_counts_5_intron / np.sum(set2_counts_5_intron))
                sreScores5_intron *= frequency_ratio_intron5 ** score_learning_rate

                psuedocount_denominator_intron3 = np.sum(set1_counts_3_intron) + np.sum(set2_counts_3_intron)
                set1_counts_3_intron = set1_counts_3_intron + np.sum(
                    set1_counts_3_intron) / psuedocount_denominator_intron3
                set2_counts_3_intron = set2_counts_3_intron + np.sum(
                    set2_counts_3_intron) / psuedocount_denominator_intron3

                frequency_ratio_intron3 = set1_counts_3_intron / np.sum(set1_counts_3_intron) / (
                        set2_counts_3_intron / np.sum(set2_counts_3_intron))
                sreScores3_intron *= frequency_ratio_intron3 ** score_learning_rate

                psuedocount_denominator_exon5 = np.sum(set1_counts_5_exon) + np.sum(set2_counts_5_exon)
                set1_counts_5_exon = set1_counts_5_exon + np.sum(set1_counts_5_exon) / psuedocount_denominator_exon5
                set2_counts_5_exon = set2_counts_5_exon + np.sum(set2_counts_5_exon) / psuedocount_denominator_exon5

                frequency_ratio_exon5 = set1_counts_5_exon / np.sum(set1_counts_5_exon) / (
                        set2_counts_5_exon / np.sum(set2_counts_5_exon))
                sreScores5_exon *= frequency_ratio_exon5 ** score_learning_rate

                psuedocount_denominator_exon3 = np.sum(set1_counts_3_exon) + np.sum(set2_counts_3_exon)
                set1_counts_3_exon = set1_counts_3_exon + np.sum(set1_counts_3_exon) / psuedocount_denominator_exon3
                set2_counts_3_exon = set2_counts_3_exon + np.sum(set2_counts_3_exon) / psuedocount_denominator_exon3

                frequency_ratio_exon3 = set1_counts_3_exon / np.sum(set1_counts_3_exon) / (
                        set2_counts_3_exon / np.sum(set2_counts_3_exon))
                sreScores3_exon *= frequency_ratio_exon3 ** score_learning_rate

                psuedocount_denominator_exon = np.sum(set1_counts_exon) + np.sum(set2_counts_exon)
                set1_counts_exon = set1_counts_exon + np.sum(set1_counts_exon) / psuedocount_denominator_exon
                set2_counts_exon = set2_counts_exon + np.sum(set2_counts_exon) / psuedocount_denominator_exon

                frequency_ratio_exon = set1_counts_exon / np.sum(set1_counts_exon) / (
                        set2_counts_exon / np.sum(set2_counts_exon))
                sreScores_exon *= frequency_ratio_exon ** score_learning_rate

                # sreScores3_intron = sreScores_intron
                # sreScores3_exon = sreScores_exon
                # sreScores5_intron = sreScores_intron
                # sreScores5_exon = sreScores_exon
                if args.split_intron == False:
                    sreScores3_intron = np.copy(sreScores_intron)
                    sreScores5_intron = np.copy(sreScores_intron)
                if args.split_exon == False:
                    sreScores3_exon = np.copy(sreScores_exon)
                    sreScores5_exon = np.copy(sreScores_exon)
        else:

            if f1 >= held_f1:  # Hold onto the scores with the highest f1 performance
                held_f1 = np.copy(f1)
                held_sreScores5_exon = np.copy(sreScores5_exon)
                held_sreScores3_exon = np.copy(sreScores3_exon)
                held_sreScores5_intron = np.copy(sreScores5_intron)
                held_sreScores3_intron = np.copy(sreScores3_intron)
                # held_rfScores_open = np.copy(rfScores_open)
                # held_rfScores_close = np.copy(rfScores_close)
            else:
                doneTime = 0  # Stop learning if the performance has decreased

            sreScores5_exon = np.copy(held_sreScores5_exon)
            sreScores3_exon = np.copy(held_sreScores3_exon)
            sreScores5_intron = np.copy(held_sreScores5_intron)
            sreScores3_intron = np.copy(held_sreScores3_intron)
            # rfScores_open = np.copy(held_rfScores_open)
            # rfScores_close = np.copy(held_rfScores_close)

        with open(os.path.join(pickle_path, "updated_multiweight_scores_step2_" + str(update_step) + ".pickle"),
                  "wb") as handle:
            pickle.dump({'sreScore_intron': sreScores_intron, 'sreScores3_intron': sreScores3_intron,
                         'sreScores5_intron': sreScores5_intron, 'sreScores_exon': sreScores_exon,
                         'sreScores3_exon': sreScores3_exon, 'sreScores5_exon': sreScores5_exon,
                         'rfScore_open': rfScores_open, 'rfScore_close': rfScores_close}, handle)
        update_step += 1
        performance.append([2, update_step, ssSens, ssPrec, f1])

    if args.use_reading_frame:
        # update open close after SREs fixed
        update_step = 0
        held_rfScores_open = np.copy(rfScores_open)
        held_rfScores_close = np.copy(rfScores_close)
        learning_counter = 0
        doneTime = 10
        while 1 < doneTime:
            learning_counter += 1
            if learning_counter > args.max_learned_scores: break
            update_scores = True

            if learning_counter % 5 == 1:
                trainGenesSub = np.copy(trainGenes1)
            elif learning_counter % 5 == 2:
                trainGenesSub = np.copy(trainGenes2)
            elif learning_counter % 5 == 3:
                trainGenesSub = np.copy(trainGenes3)
            elif learning_counter % 5 == 4:
                trainGenesSub = np.copy(trainGenes4)
            else:
                trainGenesSub = np.copy(trainGenesTest)
                update_scores = False
            if args.threads > 0:
                trainGenesSub = order_genes(trainGenesSub, args.threads, genes)
            lengths = np.array([len(str(genes[gene].seq)) for gene in trainGenesSub])
            sequences = [str(genes[gene].seq) for gene in trainGenesSub]
            coordinates_fl = np.array(
                [[np.nonzero(trueSeqs[gene] == 5)[0][0], np.nonzero(trueSeqs[gene] == 3)[0][-1]] for gene in
                 trainGenesSub])
            start_stop_gene = np.array([start_stop[gene] for gene in trainGenesSub])

            if args.use_periodic_pattern:
                exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
            else:
                exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
                exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
            intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

            for g, sequence in enumerate(sequences):
                if args.use_periodic_pattern:
                    exonic5s_all = np.array(
                        sreScores_single_phase(sequence.lower(), sreScores5_exon,
                                               kmer))
                    exonic3s_all = np.array(
                        sreScores_single_phase(sequence.lower(), sreScores3_exon,
                                               kmer))
                    for phs in [0, 1, 2]:
                        exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
                        exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
                else:
                    exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
                    exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                        np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
                intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
                intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
                    np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

            if args.use_periodic_pattern:
                pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl,
                                     pIL=pIL, pELS=pELS, pELF=pELF,
                                     pELM=pELM, pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s,
                                     exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s,
                                     intronicSREs3s=intronicSREs3s, k=kmer, sreEffect5_exon=sreEffect5_exon,
                                     sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon,
                                     sreEffect3_intron=sreEffect3_intron, start_stop=start_stop_gene, meDir=maxEntDir,
                                     rfClose=rfScores_close)
            else:
                pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                                   pELS=pELS, pELF=pELF, pELM=pELM,
                                   pELL=pELL, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                                   intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                                   sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                                   sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                                   rfOpen=rfScores_open,
                                   rfClose=rfScores_close, start_stop=start_stop_gene, meDir=maxEntDir,
                                   lmode=update_scores)  # does rf open/close need effect value? #alter viterbi!
            # Get the False Negatives and False Positives
            falsePositives = {}
            falseNegatives = {}

            num_truePositives = 0
            num_falsePositives = 0
            num_falseNegatives = 0

            for g, gene in enumerate(trainGenesSub):
                L = lengths[g]
                falsePositives[gene] = np.zeros(len(genes[gene]), dtype=int)
                falseNegatives[gene] = np.zeros(len(genes[gene]), dtype=int)

                predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
                trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

                predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
                trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

                trueThree1 = trueThrees[-1]
                trueThrees = trueThrees[:-1]

                trueFive1 = trueFives[0]
                trueFives = trueFives[1:]

                predThrees = predThrees[:-1]
                predFives = predFives[0:]

                ind_true_3 = np.where(trueThrees > start_stop[gene][0])
                ind_true_5 = np.where(trueFives < start_stop[gene][1])
                # ind_true = np.intersect1d(ind_true_3, ind_true_5)

                trueThrees = trueThrees[ind_true_3]
                trueFives = trueFives[ind_true_5]

                ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
                ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
                # int_pred = np.intersect1d(ind_pred_3, ind_pred_5)
                predThrees = predThrees[ind_pred_3]
                predFives = predFives[ind_pred_5]

                num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(
                    np.intersect1d(predFives, trueFives))

                falsePositives[gene][np.setdiff1d(predThrees, trueThrees)] = B3
                falsePositives[gene][np.setdiff1d(predFives, trueFives)] = B5
                num_falsePositives += np.sum(falsePositives[gene] > 0)

                falseNegatives[gene][np.setdiff1d(trueThrees, predThrees)] = B3
                falseNegatives[gene][np.setdiff1d(trueFives, predFives)] = B5
                num_falseNegatives += np.sum(falseNegatives[gene] > 0)

            ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
            ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
            f1 = 2 / (1 / ssSens + 1 / ssPrec)
            if update_scores:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close = get_hexamer_counts(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon,
                    sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)

                set2_counts_open += 1
                set1_counts_open += 1

                set2_counts_close += 1
                set1_counts_close += 1

                psuedocount_denominator_open = set1_counts_open + set2_counts_open
                set1_counts_open = set1_counts_open + set1_counts_open / psuedocount_denominator_open
                set2_counts_open = set2_counts_open + set2_counts_open / psuedocount_denominator_open

                psuedocount_denominator_close = set1_counts_close + set2_counts_close
                # set1_counts_close = set1_counts_close + set1_counts_close / psuedocount_denominator_close
                # set2_counts_close = set2_counts_close + set2_counts_close / psuedocount_denominator_close
                set1_counts_close = set1_counts_close / (
                            set1_counts_close + set1_counts_open)  # set1_counts_close / psuedocount_denominator_close
                set2_counts_close = set2_counts_close / (
                            set2_counts_close + set2_counts_open)  # set2_counts_close / psuedocount_denominator_close

                frequency_ratio_close = 1 / (set1_counts_close / set2_counts_close)
                rfScores_close *= frequency_ratio_close ** (score_learning_rate_rf)



            else:
                set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon, set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close = get_hexamer_counts(
                    trainGenesSub, trueSeqs, pred_all[0], falseNegatives, falsePositives, genes, kmer=kmer,
                    sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                    sreEffect3_exon=sreEffect3_exon,
                    sreEffect3_intron=sreEffect3_intron, start_stop=start_stop)

                if f1 > held_f1 or (f1 == held_f1 and (
                        set1_counts_close != 0 or set2_counts_close != 0)):  # and samef1[f1]>5):  # Hold onto the scores with the highest f1 performance
                    held_f1 = np.copy(f1)
                    # held_sreScores5_exon = np.copy(sreScores5_exon)
                    # held_sreScores3_exon = np.copy(sreScores3_exon)
                    # held_sreScores5_intron = np.copy(sreScores5_intron)
                    # held_sreScores3_intron = np.copy(sreScores3_intron)
                    held_rfScores_open = np.copy(rfScores_open)
                    held_rfScores_close = np.copy(rfScores_close)
                    if f1 not in samef1:
                        samef1[f1] = 1
                    else:
                        samef1[f1] += 1
                    if samef1[f1] == 5:
                        doneTime = 0
                else:
                    doneTime = 0  # Stop learning if the performance has decreased

                # sreScores5_exon = np.copy(held_sreScores5_exon)
                # sreScores3_exon = np.copy(held_sreScores3_exon)
                # sreScores5_intron = np.copy(held_sreScores5_intron)
                # sreScores3_intron = np.copy(held_sreScores3_intron)
                rfScores_open = np.copy(held_rfScores_open)
                rfScores_close = np.copy(held_rfScores_close)

            with open(os.path.join(pickle_path, "updated_multiweight_scores_step3_" + str(update_step) + ".pickle"),
                      "wb") as handle:
                pickle.dump({'sreScore_intron': sreScores_intron, 'sreScores3_intron': sreScores3_intron,
                             'sreScores5_intron': sreScores5_intron, 'sreScores_exon': sreScores_exon,
                             'sreScores3_exon': sreScores3_exon, 'sreScores5_exon': sreScores5_exon,
                             'rfScore_open': rfScores_open, 'rfScore_close': rfScores_close}, handle)
            update_step += 1
            performance.append([3, update_step, ssSens, ssPrec, f1])

# Save parameters
if args.save_parameters:
    para_path = os.path.join(args.out_dir, "parameters")
    os.makedirs(para_path, exist_ok=True)
    paras = {'pELF': pELF, 'pELL': pELL, 'pELM': pELM, 'pIL': pIL, 'sreScores_exon': sreScores_exon,
             'sreScores_intron': sreScores_intron, 'sreScores3_exon': sreScores3_exon,
             'sreScores5_exon': sreScores5_exon, 'sreScores3_intron': sreScores3_intron,
             'sreScores5_intron': sreScores5_intron}
    for p in paras:
        ppath = os.path.join(para_path, p + '.npy')
        np.save(ppath, paras[p])

# Filter test set
lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in testGenes])
testGenes = testGenes[lengthsOfGenes > sreEffect3_intron]

notShortIntrons = []

for gene in testGenes:
    trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

    n_fives = np.sum(trueSeqs[gene] == B5)
    n_threes = np.sum(trueSeqs[gene] == B3)

    if n_fives != n_threes:
        notShortIntrons.append(False)
    elif np.min(trueThrees - trueFives + 1) < 25:
        notShortIntrons.append(False)
    else:
        notShortIntrons.append(True)

notShortIntrons = np.array(notShortIntrons)

testGenes = testGenes[notShortIntrons]
if args.threads > 0:
    testGenes = order_genes(testGenes, args.threads, genes)
lengths = np.array([len(str(genes[gene].seq)) for gene in testGenes])
sequences = [str(genes[gene].seq) for gene in testGenes]
coordinates_fl = np.array([[np.nonzero(trueSeqs[gene] == 5)[0][0], np.nonzero(trueSeqs[gene] == 3)[0][-1]] for gene in
                           testGenes])
start_stop_gene = np.array([start_stop[gene] for gene in testGenes])

if args.use_periodic_pattern:
    exonicSREs5s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
    exonicSREs3s = np.zeros((3, len(lengths), max(lengths) - kmer + 1))
else:
    exonicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
    exonicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))
intronicSREs5s = np.zeros((len(lengths), max(lengths) - kmer + 1))
intronicSREs3s = np.zeros((len(lengths), max(lengths) - kmer + 1))

for g, sequence in enumerate(sequences):
    if args.use_periodic_pattern:
        exonic5s_all = np.array(sreScores_single_phase(sequence.lower(), sreScores5_exon, kmer))
        exonic3s_all = np.array(sreScores_single_phase(sequence.lower(), sreScores3_exon, kmer))
        for phs in [0, 1, 2]:
            exonicSREs5s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic5s_all[phs]
            exonicSREs3s[phs, g, :lengths[g] - kmer - 2 + 1] = exonic3s_all[phs]
    else:
        exonicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
            np.array(sreScores_single(sequence.lower(), sreScores5_exon, kmer)))
        exonicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
            np.array(sreScores_single(sequence.lower(), sreScores3_exon, kmer)))
    intronicSREs5s[g, :lengths[g] - kmer + 1] = np.log(
        np.array(sreScores_single(sequence.lower(), sreScores5_intron, kmer)))
    intronicSREs3s[g, :lengths[g] - kmer + 1] = np.log(
        np.array(sreScores_single(sequence.lower(), sreScores3_intron, kmer)))

if args.use_periodic_pattern:
    pred_all = viterbi_C(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL,
                         pELS=pELS,
                         pELF=pELF, pELM=pELM, pELL=pELL, phi=phi, phi5=phi5, exonicSREs5s=exonicSREs5s,
                         exonicSREs3s=exonicSREs3s, intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s,
                         k=kmer,
                         sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                         sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
                         start_stop=start_stop_gene,
                         meDir=maxEntDir, rfClose=rfScores_close)

else:
    pred_all = viterbi(sequences=sequences, transitions=transitions, coordinates_fl=coordinates_fl, pIL=pIL, pELS=pELS,
                       pELF=pELF, pELM=pELM, pELL=pELL, exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                       intronicSREs5s=intronicSREs5s, intronicSREs3s=intronicSREs3s, k=kmer,
                       sreEffect5_exon=sreEffect5_exon, sreEffect5_intron=sreEffect5_intron,
                       sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron, rfOpen=rfScores_open,
                       rfClose=rfScores_close, start_stop=start_stop_gene, meDir=maxEntDir)

# Get the Sensitivity and Precision
num_truePositives = 0
num_falsePositives = 0
num_falseNegatives = 0

falsePositives = {}
falseNegatives = {}

prediction_summary = {}
for g, gene in enumerate(testGenes):
    L = lengths[g]
    predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
    trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

    predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

    prediction_summary[gene] = {'Annotated Fives': trueFives, 'Predicted Fives': predFives,
                                'Annotated Threes': trueThrees, 'Predicted Threes': predThrees}

    if args.print_predictions:
        print(gene)
        print("\tAnnotated Fives:", trueFives, "Predicted Fives:", predFives)
        print("\tAnnotated Threes:", trueThrees, "Predicted Threes:", predThrees)

    predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
    trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

    predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]
    trueFives = np.nonzero(trueSeqs[gene] == B5)[0]

    trueThree1 = trueThrees[-1]
    trueThrees = trueThrees[:-1]

    trueFive1 = trueFives[0]
    trueFives = trueFives[1:]

    predThrees = predThrees[:-1]
    predFives = predFives[0:]

    ind_true_3 = np.where(trueThrees > start_stop[gene][0])
    ind_true_5 = np.where(trueFives < start_stop[gene][1])
    # ind_true = np.intersect1d(ind_true_3, ind_true_5)

    trueThrees = trueThrees[ind_true_3]
    trueFives = trueFives[ind_true_5]

    ind_pred_3 = np.where((predThrees > start_stop[gene][0]) & (predThrees < trueThree1))
    ind_pred_5 = np.where((predFives < start_stop[gene][1]) & (predFives > trueFive1))
    # int_pred = np.intersect1d(ind_pred_3, ind_pred_5)
    predThrees = predThrees[ind_pred_3]
    predFives = predFives[ind_pred_5]

    '''falsePositives[gene] = np.zeros(len(genes[gene]), dtype=int)
    falseNegatives[gene] = np.zeros(len(genes[gene]), dtype=int)

    falsePositives[gene][np.setdiff1d(predThrees, trueThrees)] = B3
    falsePositives[gene][np.setdiff1d(predFives, trueFives)] = B5

    falseNegatives[gene][np.setdiff1d(trueThrees, predThrees)] = B3
    falseNegatives[gene][np.setdiff1d(trueFives, predFives)] = B5'''

    num_truePositives += len(np.intersect1d(predThrees, trueThrees)) + len(np.intersect1d(predFives, trueFives))
    num_falsePositives += len(np.setdiff1d(predThrees, trueThrees)) + len(np.setdiff1d(predFives, trueFives))
    num_falseNegatives += len(np.setdiff1d(trueThrees, predThrees)) + len(np.setdiff1d(trueFives, predFives))

dfprediction = pd.DataFrame.from_dict(prediction_summary, orient='index')
dfprediction['Gene'] = dfprediction.index
dfprediction = dfprediction.melt(id_vars='Gene', var_name='Category', value_name='Location').explode('Location')
dfprediction.sort_values(['Gene', 'Category', 'Location'], ascending=[True, True, True]).to_csv(
    os.path.join(args.out_dir, "results.csv"), index=False)

if args.print_local_scores:
    local_summary = {}
    if args.use_periodic_pattern:
        scored_sequences_5, scored_sequences_3 = score_sequences_C(sequences=sequences, phi=phi,
                                                                   exonicSREs5s=exonicSREs5s, exonicSREs3s=exonicSREs3s,
                                                                   intronicSREs5s=intronicSREs5s,
                                                                   intronicSREs3s=intronicSREs3s, k=kmer,
                                                                   sreEffect5_exon=sreEffect5_exon,
                                                                   sreEffect5_intron=sreEffect5_intron,
                                                                   sreEffect3_exon=sreEffect3_exon,
                                                                   sreEffect3_intron=sreEffect3_intron, meDir=maxEntDir)

    else:
        scored_sequences_5, scored_sequences_3 = score_sequences(sequences=sequences, exonicSREs5s=exonicSREs5s,
                                                                 exonicSREs3s=exonicSREs3s,
                                                                 intronicSREs5s=intronicSREs5s,
                                                                 intronicSREs3s=intronicSREs3s, k=kmer,
                                                                 sreEffect5_exon=sreEffect5_exon,
                                                                 sreEffect5_intron=sreEffect5_intron,
                                                                 sreEffect3_exon=sreEffect3_exon,
                                                                 sreEffect3_intron=sreEffect3_intron, meDir=maxEntDir)

    for g, gene in enumerate(testGenes):
        L = lengths[g]
        predThrees = np.nonzero(pred_all[0][g, :L] == 3)[0]
        predFives = np.nonzero(pred_all[0][g, :L] == 5)[0]

        local_summary[gene] = {'Internal Exons': [], 'Introns': []}
        # print(gene, "Internal Exons:")
        for i, three in enumerate(predThrees[:-1]):
            five = predFives[i + 1]
            # print("\t", np.log2(scored_sequences_5[g,five]) + np.log2(scored_sequences_3[g,three]) + np.log2(pELM[five - three]))
            local_summary[gene]['Internal Exons'].append(
                np.log2(scored_sequences_5[g, five]) + np.log2(scored_sequences_3[g, three]) + np.log2(
                    pELM[five - three]))

        # print(gene, "Introns:")
        for i, three in enumerate(predThrees):
            five = predFives[i]
            # print("\t", np.log2(scored_sequences_5[g,five]) + np.log2(scored_sequences_3[g,three]) + np.log2(pELM[-five + three]))
            local_summary[gene]['Introns'].append(
                np.log2(scored_sequences_5[g, five]) + np.log2(scored_sequences_3[g, three]) + np.log2(
                    pELM[-five + three]))
    dflocal = pd.DataFrame.from_dict(local_summary, orient='index')
    dflocal.to_csv(os.path.join(args.out_dir, "local_scores.csv"))

ssSens = num_truePositives / (num_truePositives + num_falseNegatives)
ssPrec = num_truePositives / (num_truePositives + num_falsePositives)
f1 = 2 / (1 / ssSens + 1 / ssPrec)
performance.append([4, 0, ssSens, ssPrec, f1])
dfperformance = pd.DataFrame(performance, columns=['step', 'update_step', 'sensitivity', 'precision', 'f1'])
dfperformance.to_csv(os.path.join(args.out_dir, "performance.csv"), index=False)
print("Final Test Metrics", "Recall", ssSens, "Precision", ssPrec, "f1", f1)
