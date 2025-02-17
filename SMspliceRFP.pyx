#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
from math import exp 
from libc.math cimport exp as c_exp



def baseToInt(str base):
    if base == 'a': return 0
    elif base == 'c': return 1
    elif base == 'g': return 2
    elif base == 't': return 3
    else:
        print("nonstandard base encountered:", base)
        return -1

def intToBase(int i):
    if i == 0: return 'a'
    elif i == 1: return 'c'
    elif i == 2: return 'g'
    elif i == 3: return 't'
    else: 
        print("nonbase integer encountered:", i)
        return ''

def hashSequence(str seq):
    cdef int i
    cdef int sum = 0 
    cdef int l = len(seq)
    for i in range(l):
        sum += (4**(l-i-1))*baseToInt(seq[i])
    return sum
    
def unhashSequence(int num, int l):
    seq = ''
    for i in range(l):
        seq += intToBase(num // 4**(l-i-1))
        num -= (num // 4**(l-i-1))*(4**(l-i-1))
    return seq

def stopCodonCount(seq):
    # Cumulative counts of stop codons along the sequence
    stop_codon = ['TAA', 'TAG', 'TGA']
    rf = np.zeros((3, len(seq)))
    for i in range(len(seq)):
        if i != 0:
            rf[:, i] = rf[:, i - 1]
        if seq[i:i + 3].upper() in stop_codon:
            rf[i % 3, i] += 1
    return rf

def trueSequencesCannonical(genes, annotations, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys(): 
            print(gene, 'has annotation, but was not found in the fasta file of genes') 
            continue
        
        transcript = annotations[gene]
        if len(transcript) == 1: 
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype = int) + E
            continue # skip the rest for a single exon case

        # First exon
        true = np.zeros(len(genes[gene]), dtype = int) + I
        three = transcript[0][0] - 1 # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three+1, five)] = E
        true[five] = B5
        '''if gene=='Oaz1':
            print(five)'''
        
        # Internal exons 
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three+1, five)] = E
            '''if gene == 'Oaz1':
                print(five,three)'''
            
        # Last exon 
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1 # Marking the end of the last exon
        true[range(three+1, five)] = E
        '''if gene=='Oaz1':
            print(three)'''
                
        trueSeqs[gene] = true
        
    return(trueSeqs)

def trueSequencesCannonical_C(genes, annotations, phases, E = 0, I = 1, B3 = 3, B5 = 5):
    # Converts gene annotations to sequences of integers indicating whether the sequence is exonic, intronic, or splice site,
    # Inputs
    #   - genes: a biopython style dictionary of the gene sequences
    #   - annotations: the splicing annotations dictionary
    #   - E, I, B3, B5: the integer indicators for exon, intron, 3'ss, and 5'ss, respectively
    trueSeqs = {}
    phaseSeqs = {}
    for gene in annotations.keys():
        if gene not in genes.keys():
            print(gene, 'has annotation, but was not found in the fasta file of genes')
            continue

        transcript = annotations[gene]
        if len(transcript) == 1:
            trueSeqs[gene] = np.zeros(len(genes[gene]), dtype=int) + E
            continue  # skip the rest for a single exon case

        # First exon
        countexon = 0
        true = np.zeros(len(genes[gene]), dtype=int) + I
        phase = np.zeros(len(genes[gene]), dtype=int) + I
        three = transcript[0][0] - 1  # Marking the beginning of the first exon
        five = transcript[0][1] + 1
        true[range(three + 1, five)] = E
        #print(phases[gene][countexon], gene, len(phases[gene][countexon]))
        #print(gene)
        phase[range(three + 1, five)] = [x + 10 for x in phases[gene][countexon]]
        countexon += 1
        true[five] = B5
        #?how to deal with first exon with 5'UTR?

        #print(phases[gene],countexon)
        # Internal exons
        for exon in transcript[1:-1]:
            three = exon[0] - 1
            five = exon[1] + 1
            true[three] = B3
            true[five] = B5
            true[range(three + 1, five)] = E
            #print(gene)
            phase[range(three + 1, five)] = [x + 10 for x in phases[gene][countexon]]
            countexon += 1

        # Last exon
        three = transcript[-1][0] - 1
        true[three] = B3
        five = transcript[-1][1] + 1  # Marking the end of the last exon
        true[range(three + 1, five)] = E
        phase[range(three + 1, five)] = [x + 10 for x in phases[gene][countexon]]
        countexon += 1

        trueSeqs[gene] = true
        phaseSeqs[gene] = phase

    return (trueSeqs, phaseSeqs)

def trainAllTriplets(sequences, cutoff = 10**(-5)):
    # Train maximum entropy models from input sequences with triplet conditions
    train = np.zeros((len(sequences),len(sequences[0])), dtype = int)
    for (i, seq) in enumerate(sequences):
        for j in range(len(seq)):
            train[i,j] = baseToInt(seq[j])
    prob = np.log(np.zeros(4**len(sequences[0])) + 4**(-len(sequences[0])))
    Hprev = -np.sum(prob*np.exp(prob))/np.log(2)
    H = -1
    sequences = np.zeros((4**len(sequences[0]),len(sequences[0])), dtype = int)
    l = len(sequences[0]) - 1 
    for i in range(sequences.shape[1]):
        sequences[:,i] = ([0]*4**(l-i) + [1]*4**(l-i) + [2]*4**(l-i) +[3]*4**(l-i))*4**i
    while np.abs(Hprev - H) > cutoff:
        #print(np.abs(Hprev - H))
        Hprev = H
        for pos in range(sequences.shape[1]):
            for base in range(4):
                Q = np.sum(train[:,pos] == base)/float(train.shape[0])
                if Q == 0: continue
                Qhat = np.sum(np.exp(prob[sequences[:,pos] == base]))
                prob[sequences[:,pos] == base] += np.log(Q) - np.log(Qhat)
                prob[sequences[:,pos] != base] += np.log(1-Q) - np.log(1-Qhat)
                
                for pos2 in np.setdiff1d(range(sequences.shape[1]), range(pos+1)):
                    for base2 in range(4):
                        Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2))/float(train.shape[0])
                        if Q == 0: continue
                        which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)
                        Qhat = np.sum(np.exp(prob[which]))
                        prob[which] += np.log(Q) - np.log(Qhat)
                        prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
                        
                        for pos3 in np.setdiff1d(range(sequences.shape[1]), range(pos2+1)):
                            for base3 in range(4):
                                Q = np.sum((train[:,pos] == base)*(train[:,pos2] == base2)*(train[:,pos3] == base3))/float(train.shape[0])
                                if Q == 0: continue
                                which = (sequences[:,pos] == base)*(sequences[:,pos2] == base2)*(sequences[:,pos3] == base3)
                                Qhat = np.sum(np.exp(prob[which]))
                                prob[which] += np.log(Q) - np.log(Qhat)
                                prob[np.invert(which)] += np.log(1-Q) - np.log(1-Qhat)
        H = -np.sum(prob*np.exp(prob))/np.log(2)
    return np.exp(prob)

def structuralParameters(genes, annotations, minIL = 0):
    # Get the empirical length distributions for introns and single, first, middle, and last exons, as well as number exons per gene
    
    # Transitions
    numExonsPerGene = [] 
    
    # Length Distributions
    lengthSingleExons = []
    lengthFirstExons = []
    lengthMiddleExons = []
    lengthLastExons = []
    lengthIntrons = []
    
    for gene in genes:
        if len(annotations[gene]) == 0: 
            print('missing annotation for', gene)
            continue
        numExons = 0
        introns = []
        singleExons = []
        firstExons = []
        middleExons = []
        lastExons = []
        
        for transcript in annotations[gene].values():
            numExons += len(transcript)
            
            # First exon 
            three = transcript[0][0] # Make three the first base
            five = transcript[0][1] + 1
            if len(transcript) == 1: 
                singleExons.append((three, five-1))
                continue # skip the rest for a single exon case
            firstExons.append((three, five-1)) # since three is the first base
            
            # Internal exons 
            for exon in transcript[1:-1]:
                three = exon[0] - 1 
                introns.append((five+1,three-1))
                five = exon[1] + 1
                middleExons.append((three+1, five-1))
                
            # Last exon 
            three = transcript[-1][0] - 1
            introns.append((five+1,three-1))
            five = transcript[-1][1] + 1
            lastExons.append((three+1, five-1))
        
        geneIntronLengths = [minIL]
        for intron in set(introns):
            geneIntronLengths.append(intron[1] - intron[0] + 1)
        
        if np.min(geneIntronLengths) < minIL: continue
        
        for intron in set(introns): lengthIntrons.append(intron[1] - intron[0] + 1)
        for exon in set(singleExons): lengthSingleExons.append(exon[1] - exon[0] + 1)
        for exon in set(firstExons): lengthFirstExons.append(exon[1] - exon[0] + 1)
        for exon in set(middleExons): lengthMiddleExons.append(exon[1] - exon[0] + 1)
        for exon in set(lastExons): lengthLastExons.append(exon[1] - exon[0] + 1)
            
        numExonsPerGene.append(float(numExons)/len(annotations[gene]))
        
    return(numExonsPerGene, lengthSingleExons, lengthFirstExons, lengthMiddleExons, lengthLastExons, lengthIntrons)

def adaptive_kde_tailed(lengths, N, geometric_cutoff = .8, lower_cutoff=0):
    adaptive_kde = GaussianKDE(alpha = 1) 
    adaptive_kde.fit(np.array(lengths)[:,None]) 
    
    lengths = np.array(lengths)
    join = np.sort(lengths)[int(len(lengths)*geometric_cutoff)] 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = adaptive_kde.predict(np.arange(join+1)[:,None])
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)
    
def geometric_smooth_tailed(lengths, N, bandwidth, join, lower_cutoff=0):
    lengths = np.array(lengths)
    smoothing = KernelDensity(bandwidth = bandwidth).fit(lengths[:, np.newaxis]) 
    
    smoothed = np.zeros(N)
    smoothed[:join+1] = np.exp(smoothing.score_samples(np.arange(join+1)[:,None]))
    
    s = 1-np.sum(smoothed)
    p = smoothed[join]
    smoothed[join+1:] = np.exp(np.log(p) + np.log(s/(s+p))*np.arange(1,len(smoothed)-join))
    smoothed[:lower_cutoff] = 0
    smoothed /= np.sum(smoothed)
    
    return(smoothed)

def maxEnt5(geneNames, genes, dir):
    # Get all the 5'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob = np.load(dir + '/maxEnt5_prob.npy')
    prob0 = np.load(dir + '/maxEnt5_prob0.npy') 
        
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequence5 = np.array([hashSequence(sequence[i:i+9]) for i in range(len(sequence)-9+1)])
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt5_single(str seq, str dir):
    prob = np.load(dir + 'maxEnt5_prob.npy')
    prob0 = np.load(dir + 'maxEnt5_prob0.npy')
    
    seq = seq.lower()
    sequence5 = np.array([hashSequence(seq[i:i+9]) for i in range(len(seq)-9+1)])
    scores = np.log2(np.zeros(len(seq)))
    scores[3:-5] = np.log2(prob[sequence5]) - np.log2(prob0[sequence5])
    return np.exp2(scores)
    
def maxEnt3(geneNames, genes, dir):
    # Get all the 3'SS maxent scores for each of the genes in geneNames
    scores = {}
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    for gene in geneNames:
        sequence = str(genes[gene].seq).lower()
        sequences23 = [sequence[i:i+23] for i in range(len(sequence)-23+1)]
        hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
        hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
        hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
        hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
        hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
        hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
        hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
        hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
        hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
        
        probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
        scores[gene] = np.zeros(len(sequence)) - np.inf
        scores[gene][19:-3] = probs
        scores[gene] = np.exp2(scores[gene])
    
    return scores
    
def maxEnt3_single(str seq, str dir):
    prob0 = np.load(dir + 'maxEnt3_prob0.npy')
    prob1 = np.load(dir + 'maxEnt3_prob1.npy')
    prob2 = np.load(dir + 'maxEnt3_prob2.npy')
    prob3 = np.load(dir + 'maxEnt3_prob3.npy')
    prob4 = np.load(dir + 'maxEnt3_prob4.npy')
    prob5 = np.load(dir + 'maxEnt3_prob5.npy')
    prob6 = np.load(dir + 'maxEnt3_prob6.npy')
    prob7 = np.load(dir + 'maxEnt3_prob7.npy')
    prob8 = np.load(dir + 'maxEnt3_prob8.npy')
    
    prob0_0 = np.load(dir + 'maxEnt3_prob0_0.npy')
    prob1_0 = np.load(dir + 'maxEnt3_prob1_0.npy')
    prob2_0 = np.load(dir + 'maxEnt3_prob2_0.npy')
    prob3_0 = np.load(dir + 'maxEnt3_prob3_0.npy')
    prob4_0 = np.load(dir + 'maxEnt3_prob4_0.npy')
    prob5_0 = np.load(dir + 'maxEnt3_prob5_0.npy')
    prob6_0 = np.load(dir + 'maxEnt3_prob6_0.npy')
    prob7_0 = np.load(dir + 'maxEnt3_prob7_0.npy')
    prob8_0 = np.load(dir + 'maxEnt3_prob8_0.npy')
    
    seq = seq.lower()
    sequences23 = [seq[i:i+23] for i in range(len(seq)-23+1)]
    hash0 = np.array([hashSequence(seq[0:7]) for seq in sequences23])
    hash1 = np.array([hashSequence(seq[7:14]) for seq in sequences23])
    hash2 = np.array([hashSequence(seq[14:]) for seq in sequences23])
    hash3 = np.array([hashSequence(seq[4:11]) for seq in sequences23])
    hash4 = np.array([hashSequence(seq[11:18]) for seq in sequences23])
    hash5 = np.array([hashSequence(seq[4:7]) for seq in sequences23])
    hash6 = np.array([hashSequence(seq[7:11]) for seq in sequences23])
    hash7 = np.array([hashSequence(seq[11:14]) for seq in sequences23])
    hash8 = np.array([hashSequence(seq[14:18]) for seq in sequences23])
    
    probs = np.log2(prob0[hash0]) + np.log2(prob1[hash1]) + np.log2(prob2[hash2]) + \
            np.log2(prob3[hash3]) + np.log2(prob4[hash4]) - np.log2(prob5[hash5]) - \
            np.log2(prob6[hash6]) - np.log2(prob7[hash7]) - np.log2(prob8[hash8]) - \
            (np.log2(prob0_0[hash0]) + np.log2(prob1_0[hash1]) + np.log2(prob2_0[hash2]) + \
            np.log2(prob3_0[hash3]) + np.log2(prob4_0[hash4]) - np.log2(prob5_0[hash5]) - \
            np.log2(prob6_0[hash6]) - np.log2(prob7_0[hash7]) - np.log2(prob8_0[hash8]))
            
    scores = np.log2(np.zeros(len(seq)))
    scores[19:-3] = probs
    return np.exp2(scores)

def sreScores_single(str seq, double [:] sreScores, int kmer = 6):
    indices = [hashSequence(seq[i:i+kmer]) for i in range(len(seq)-kmer+1)]
    sequenceSRES = [sreScores[indices[i]] for i in range(len(indices))]
    return sequenceSRES

def sreScores_single_phase(sresequence, sreScoref, kmer):
    sequence = np.array([hashSequence(sresequence[i:i + kmer]) for i in range(len(sresequence) - kmer + 1)])
    #f1,f2,f3=[],[],[]
    fscores=np.zeros((3,len(sequence)-2))
    for i in range(len(sequence)-2):
        #f1.append(np.log(sreScoref[0,sequence[i]]*sreScoref[1,sequence[i+1]]*sreScoref[2,sequence[i+2]]))
        fscores[0,i]+=(np.log(sreScoref[0,sequence[i]])+np.log(sreScoref[1,sequence[i+1]])+np.log(sreScoref[2,sequence[i+2]]))
        #f2.append(np.log(sreScoref[1,sequence[i]] * sreScoref[2,sequence[i + 1]] * sreScoref[0,sequence[i + 2]]))
        fscores[1,i]+=(np.log(sreScoref[1,sequence[i]])+np.log(sreScoref[2,sequence[i + 1]])+np.log(sreScoref[0,sequence[i + 2]]))
        #f3.append(np.log(sreScoref[2,sequence[i]] * sreScoref[0,sequence[i + 1]] * sreScoref[1,sequence[i + 2]]))
        fscores[2,i]+=(np.log(sreScoref[2,sequence[i]])+np.log(sreScoref[0,sequence[i + 1]])+np.log(sreScoref[1,sequence[i + 2]]))
    return fscores#np.array([np.array(f1),np.array(f2),np.array(f3)])

def get_all_5ss(gene, reference, genes):
    # Get all the 5'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonStarts[i-1] + 2 for i in range(len(exonStarts),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonEnds[i] - start + 1 for i in range(len(exonEnds))]
        
    return(annnotation)

def get_all_3ss(gene, reference, genes):
    # Get all the 3'SS for a gene based on the annotation in reference
    info = genes[gene].description.split(' ')
    if reference.loc[gene,6] == ',': exonEnds = []
    else: exonEnds = [int(start)-1 for start in reference.loc[gene,6].split(',')[:-1]] # intron starts -> exon ends
    if reference.loc[gene,7] == ',': exonStarts = []
    else: exonStarts = [int(end)+1 for end in reference.loc[gene,7].split(',')[:-1]] # exon starts -> intron ends
    
    if info[6] == 'Strand:-': 
        stop = int(info[5][5:])
        annnotation = [stop - exonEnds[i-1] - 2 for i in range(len(exonEnds),0,-1)]      
        
    elif info[6] == 'Strand:+': 
        start = int(info[4][6:])
        annnotation = [exonStarts[i] - start - 3 for i in range(len(exonStarts))]
        
    return(annnotation)

def get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, start_stop, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    true_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4**kmer, dtype = np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4**kmer, dtype = np.dtype("i"))

    set1_counts_open = 0
    set1_counts_close = 0
    set2_counts_open = 0
    set2_counts_close = 0
    #exons are between 3' and 5'

    '''for gene in geneNames:
        for i in range(len(decoySS[gene])):
            if decoySS[gene][i] == 3:
                # Start looking for the next 5 after the current 3
                for j in range(i + 1, len(decoySS[gene])):
                    if decoySS[gene][j] == 3:
                        break  # Stop if another 3 is encountered before finding a 5
                    if decoySS[gene][j] == 5:
                        #pairs.append((i, j))  # Store the pair of indices
                        exonseq = genes[gene].seq[i + 1:j]
                        if (j-i-1)<50 or (j-i-1)>250:
                            break
                        stopsum = (stopCodonCount(exonseq).sum(1)>0).sum()
                        addopen = (stopsum <3)
                        addclose = (stopsum ==3)
                        set2_counts_open += addopen
                        set2_counts_close += addclose
                        #print('decoy:',genes[gene].seq[i + 1-2:j+2],addopen,stopsum)
                        break  # Move on to the next 3'''
    
    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]#[:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]#[1:]
        trueFive1=start_stop[gene][0]#trueFives[0]
        trueThree1=start_stop[gene][1]#trueThrees[-1]
        trueFives=trueFives[1:]
        #trueFives=np.nonzero(trueFives > start_stop[gene][0])[0]#
        trueThrees=trueThrees[:-1]
        #trueThrees=np.nonzero(trueThrees > start_stop[gene][1])[0]#
        '''print(len(trueSeqs))
        print(np.nonzero(trueSeqs[gene] == B3)[0])
        print(np.nonzero(trueSeqs[gene] == B5)[0])'''

        for i in range(len(trueThrees)):
            #print(trueThrees.shape,trueFives.shape,gene)
            #print(trueSeqs[gene])
            three = trueThrees[i]
            five = trueFives[i]

            if three<trueFive1 or five >trueThree1:
                continue

            exonseq = genes[gene].seq[three + 1:five]
            if 50<=five-three<=250:
                stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
                addopen = (stopsum < 3)
                addclose = (stopsum == 0)
                set1_counts_open += addopen
                set1_counts_close += addclose
            
            # 3'SS
            sequence = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[three+4:].lower())
            if five-3 < three+sreEffect3_exon+1: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[three-sreEffect3_intron:three-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_3_intron[s] += 1
                
            # 5'SS
            sequence = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:five-3].lower())
            if five-sreEffect5_exon < three+4: sequence = str(genes[gene].seq[three+4:five-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[five+6:five+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: true_counts_5_intron[s] += 1
        
        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            #corresponding3=decoyThrees[np.where(decoyFives == ss)[0][0]-1]
            if ss < trueFive1 or ss > trueThree1:
                continue
            sequence = str(genes[gene].seq[ss-sreEffect5_exon:ss-3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss+6:ss+sreEffect5_intron+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_5_intron[s] += 1
    
        for ss in decoyThrees:
            #print(decoyFives,decoyThrees,ss)
            #corresponding5=decoyFives[np.where(decoyThrees == ss)[0][0]+1]
            if ss < trueFive1 or ss > trueThree1:
                continue
            for j in range(ss + 1, len(decoySS[gene])):
                if decoySS[gene][j] == 3:
                    break  # Stop if another 3 is encountered before finding a 5
                if decoySS[gene][j] == 5:
                    #pairs.append((i, j))  # Store the pair of indices
                    exonseq = genes[gene].seq[ss + 1:j]
                    if (j - ss - 1) < 50 or (j - ss - 1) > 250:
                        break
                    stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
                    addopen = (stopsum < 3)
                    addclose = (stopsum == 3)
                    set2_counts_open += addopen
                    set2_counts_close += addclose
                    #print('decoy:',genes[gene].seq[i + 1-2:j+2],addopen,stopsum)
                    break  # Move on to the next 3

            sequence = str(genes[gene].seq[ss+4:ss+sreEffect3_exon+1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss+4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_exon[s] += 1
            
            sequence = str(genes[gene].seq[ss-sreEffect3_intron:ss-19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss-19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i+kmer]) for i in range(len(sequence)-kmer+1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    print(set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close)
    return(true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, 
           decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon,
            set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close)

def get_hexamer_real_decoy_counts_C(geneNames, trueSeqs, decoySS, genes, phaseSeqs, kmer, sreEffect5_exon,
                                  sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, start_stop, B3 = 3, B5 = 5):
    # Get the counts of hexamers in the flanking regions for real and decoy ss with restriction to exons and introns for the real ss
    true_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    true_counts_5_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    true_counts_5_exon_phase = np.zeros((3, 4 ** kmer), dtype=np.dtype("i"))
    true_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    true_counts_3_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    true_counts_3_exon_phase = np.zeros((3, 4 ** kmer), dtype=np.dtype("i"))
    decoy_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    decoy_counts_5_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    decoy_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    decoy_counts_3_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))

    phi = np.zeros(3, dtype=np.dtype("i"))  #frequency of phases in the training set
    phi5 = np.zeros(3, dtype=np.dtype("i"))

    for gene in geneNames:
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]  #[:-1]
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]  #[1:]
        trueFive1 = start_stop[gene][0]
        trueThree1 = start_stop[gene][1]
        trueFives = trueFives[1:]
        trueThrees = trueThrees[:-1]

        for i in range(len(trueThrees)):
            #print(trueThrees.shape,trueFives.shape,gene)
            #print(trueSeqs[gene])
            three = trueThrees[i]
            five = trueFives[i]
            if three < trueFive1 or five > trueThree1:
                continue
            phase = phaseSeqs[gene][three + 4] - 10
            phase2=phaseSeqs[gene][three+sreEffect3_exon]-10
            #print(phase,phase2)

            # 3'SS
            countphase = True
            lastbase = three + sreEffect3_exon if len(phaseSeqs[gene]) >= (three + sreEffect3_exon + 1) else len(
                phaseSeqs[gene]) - 1
            endphase = phaseSeqs[gene][lastbase] - 10
            if phase2 not in [0, 1, 2]:# or phase2 not in [0,1,2]:
                countphase = False
                '''for i in range(lastbase,three+3,-1):
                    if phaseSeqs[gene][i] >= 10:
                        sequence = str(genes[gene].seq[three+4:i+1].lower())
                        countphase=True
                        break'''
            sequence = str(genes[gene].seq[three + 4:three + sreEffect3_exon + 1].lower())
            '''oriseq = str(genes[gene].seq[three+4:three+sreEffect3_exon+1].lower())
            if len(sequence) == 0:
                sequence = str(genes[gene].seq[three+4:].lower())
                countphase=False
            if five-3 < three+sreEffect3_exon+1:
                sequence = str(genes[gene].seq[three+4:five-3].lower())
                countphase=False
            if len(sequence) < kmer: continue'''
            if countphase:  #len(sequence)==sreEffect3_exon-3 and endphase in [0,1,2]:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                #for s in sequence: true_counts_3_exon[s] += 1
                #if countphase:
                phi[phase] += 1
                for sidx in range(len(sequence)): true_counts_3_exon_phase[(phase + sidx) % 3, sequence[sidx]] += 1

            sequence = str(genes[gene].seq[three - sreEffect3_intron:three - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:three - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: true_counts_3_intron[s] += 1

            # 5'SS
            countphase = True
            phase = phaseSeqs[gene][
                        five - sreEffect5_exon] - 10  #[five-sreEffect5_exon]-10#phase+five-sreEffect5_exon-(three+1))%3
            phase2 = phaseSeqs[gene][five - 2] - 10
            phase_ = (phaseSeqs[gene][five - 2] - 10 - (sreEffect5_exon - 2)) % 3
            #firstphase=phaseSeqs[gene][five-sreEffect5_exon]-10
            if phase2 not in [0, 1, 2]:# or phase not in [0,1,2]:
                countphase = False
                f'''or i in range(five - sreEffect5_exon, five - 3):
                    if phaseSeqs[gene][i] >= 10:
                        sequence = str(genes[gene].seq[i:five - 3].lower())
                        phase = phaseSeqs[gene][i] - 10
                        countphase = True
                        break'''
            sequence = str(genes[gene].seq[five - sreEffect5_exon:five - 3].lower())
            #oriseq = str(genes[gene].seq[five-sreEffect5_exon:five-3].lower())
            '''if len(sequence) == 0:
                sequence = str(genes[gene].seq[:five-3].lower())
                countphase=False
                #phase_=phaseSeqs[gene][0]-1#phase=phaseSeqs[gene][0]-1
            if five-sreEffect5_exon < three+4:
                sequence = str(genes[gene].seq[three+4:five-3].lower())
                countphase=False
                #phase_=phase
            if len(sequence) < kmer: continue'''
            if countphase:  #len(sequence)==sreEffect5_exon-3 and firstphase in [0,1,2]:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])

                for s in sequence: true_counts_5_exon[s] += 1
                #if countphase:
                phi5[phase_] += 1
                for sidx in range(len(sequence)): true_counts_5_exon_phase[(phase_ + sidx) % 3, sequence[sidx]] += 1

            sequence = str(genes[gene].seq[five + 6:five + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[five + 6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: true_counts_5_intron[s] += 1

        decoyThrees = np.nonzero(decoySS[gene] == B3)[0]
        decoyFives = np.nonzero(decoySS[gene] == B5)[0]
        for ss in decoyFives:
            if ss < trueFive1 or ss > trueThree1:
                continue
            sequence = str(genes[gene].seq[ss - sreEffect5_exon:ss - 3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 3].lower())
            if len(sequence) < kmer: continue
            if len(sequence) == sreEffect5_exon - 3:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                for s in sequence: decoy_counts_5_exon[s] += 1

            sequence = str(genes[gene].seq[ss + 6:ss + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: decoy_counts_5_intron[s] += 1

        for ss in decoyThrees:
            if ss < trueFive1 or ss > trueThree1:
                continue
            sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 4:].lower())
            if len(sequence) < kmer: continue
            if len(sequence) == sreEffect5_exon - 3:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                for s in sequence: decoy_counts_3_exon[s] += 1

            sequence = str(genes[gene].seq[ss - sreEffect3_intron:ss - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: decoy_counts_3_intron[s] += 1
    #phi=phi+np.sum(phi)
    return (true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon,
            decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon,
            true_counts_5_exon_phase, true_counts_3_exon_phase, phi / np.sum(phi), phi5 / np.sum(phi5))  #,
    #set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close)

def get_hexamer_counts(geneNames, trueSeqs, pred_all, set1, set2, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon,
                       sreEffect3_intron, start_stop, B3=3, B5=5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set1_counts_5_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set1_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set1_counts_3_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_5_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_3_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))

    set1_counts_open=0
    set1_counts_close=0
    set2_counts_open=0
    set2_counts_close=0
    #exons are between 3' and 5'

    for g, gene in enumerate(geneNames):
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]
        trueFive1 = start_stop[gene][0]#np.nonzero(trueSeqs[gene] == B5)[0][0]
        trueThree1 = start_stop[gene][1]#np.nonzero(trueSeqs[gene] == B3)[0][-1]
        predThrees = np.nonzero(pred_all[g,] == B3)[0]
        predFives = np.nonzero(pred_all[g,] == B5)[0]

        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]


        for ss in set1Fives:
            if ss==trueFives[0]:
                continue
            corresponding3=trueThrees[np.where(trueFives == ss)[0][0]-1]
            if corresponding3<trueFive1 or ss>trueThree1:
                continue
            exonseq = genes[gene].seq[corresponding3+1:ss]
            stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
            addopen = (stopsum < 3)
            addclose = (stopsum == 3)
            set1_counts_open += addopen
            set1_counts_close += addclose
            sequence = str(genes[gene].seq[ss - sreEffect5_exon:ss - 3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_5_exon[s] += 1

            sequence = str(genes[gene].seq[ss + 6:ss + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_5_intron[s] += 1

        for ss in set1Threes:
            if ss==trueThrees[-1]:
                continue
            corresponding5=trueFives[np.where(trueThrees == ss)[0][0]+1]
            if ss<trueFive1 or corresponding5>trueThree1:
                continue
            if corresponding5 not in set1Fives:
                exonseq = genes[gene].seq[ss + 1:corresponding5]
                stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
                addopen = (stopsum < 3)
                addclose = (stopsum == 3)
                set1_counts_open += addopen
                set1_counts_close += addclose
            sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_3_exon[s] += 1

            sequence = str(genes[gene].seq[ss - sreEffect3_intron:ss - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_3_intron[s] += 1

        for ss in set2Fives:
            if ss==predFives[0]:
                continue
            corresponding3=predThrees[np.where(predFives == ss)[0][0]-1]
            if corresponding3<trueFive1 or ss>trueThree1:
                continue
            exonseq = genes[gene].seq[corresponding3+1:ss]
            stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
            addopen = (stopsum < 3)
            addclose = (stopsum == 3)
            set2_counts_open += addopen
            set2_counts_close += addclose
            sequence = str(genes[gene].seq[ss - sreEffect5_exon:ss - 3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 3].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_5_exon[s] += 1

            sequence = str(genes[gene].seq[ss + 6:ss + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_5_intron[s] += 1

        for ss in set2Threes:
            if ss==predThrees[-1]:
                continue
            corresponding5 = predFives[np.where(predThrees == ss)[0][0]+1]
            if ss<trueFive1 or corresponding5>trueThree1:
                continue
            if corresponding5 not in set2Fives:
                exonseq = genes[gene].seq[ss + 1:corresponding5]
                stopsum = (stopCodonCount(exonseq).sum(1) > 0).sum()
                addopen = (stopsum < 3)
                addclose = (stopsum == 3)
                set2_counts_open += addopen
                set2_counts_close += addclose
                '''if addclose:
                    print(gene,ss, corresponding5,trueFive1,trueThree1)
                    print(trueFives)
                    print(trueThrees)'''
            sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 4:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_3_exon[s] += 1

            sequence = str(genes[gene].seq[ss - sreEffect3_intron:ss - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_3_intron[s] += 1

    return (set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon,
            set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon,
            set1_counts_open, set1_counts_close, set2_counts_open, set2_counts_close)

def get_hexamer_counts_C(geneNames, trueSeqs, pred_all, set1, set2, genes, phaseSeqs, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon,
                       sreEffect3_intron, start_stop, B3=3, B5=5):
    # Get the counts of hexamers in the flanking regions for two sets of ss
    set1_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set1_counts_5_exon = np.zeros((3,4 ** kmer), dtype=np.dtype("i"))
    set1_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set1_counts_3_exon = np.zeros((3,4 ** kmer), dtype=np.dtype("i"))
    set2_counts_5_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_5_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_3_intron = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    set2_counts_3_exon = np.zeros(4 ** kmer, dtype=np.dtype("i"))
    phicount=np.zeros((2,4))
    fpcount=np.zeros(2)


    for g,gene in enumerate(geneNames):
        trueFives = np.nonzero(trueSeqs[gene] == B5)[0]
        trueThrees = np.nonzero(trueSeqs[gene] == B3)[0]

        set1Threes = np.nonzero(set1[gene] == B3)[0]
        set1Fives = np.nonzero(set1[gene] == B5)[0]
        set2Threes = np.nonzero(set2[gene] == B3)[0]
        set2Fives = np.nonzero(set2[gene] == B5)[0]

        trueFive1 = start_stop[gene][0]#np.nonzero(trueSeqs[gene] == B5)[0][0]
        trueThree1 = start_stop[gene][1]#np.nonzero(trueSeqs[gene] == B3)[0][-1]

        predThrees = np.nonzero(pred_all[g,] == B3)[0]
        predFives = np.nonzero(pred_all[g,] == B5)[0]


        for ss in set1Fives:
            if ss==trueFives[0]:
                continue
            corresponding3 = trueThrees[np.where(trueFives == ss)[0][0] - 1]
            if corresponding3 < trueFive1 or ss > trueThree1:
                continue
            sequence = str(genes[gene].seq[ss - sreEffect5_exon:ss - 3].lower())
            phase_ = phaseSeqs[gene][ss - sreEffect5_exon] - 10#[ss-2]-10#[ss - sreEffect5_exon] - 10
            #firstphase=phaseSeqs[gene][ss-sreEffect5_exon]-10
            phase2 = phaseSeqs[gene][ss - 3] - 10
            phase=(phaseSeqs[gene][ss - 3] - 10 - (sreEffect5_exon+3))%3
            count=True
            if phase2 not in [0,1,2]:# or phase2 not in [0,1,2]:
                #print(phase,phase2,corresponding3,ss)

                count = False
                phicount[0, 3] +=1
                '''for i in range(ss-sreEffect5_exon,ss-3):
                    if phaseSeqs[gene][i]>=10:
                        sequence=str(genes[gene].seq[i:ss-3].lower())
                        phase=phaseSeqs[gene][i]-10
                        phicount[0, phase] += 1
                        count=True
                        break
                if not count:
                    phicount[0, 3] += 1'''
            '''if len(sequence) == 0:
                sequence = str(genes[gene].seq[:ss - 3].lower())
                #phase = phaseSeqs[gene][0]-1
                count=False
            if len(sequence) < kmer: continue'''
            if count:#len(sequence)==sreEffect5_exon-3 and firstphase in [0,1,2]:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                #if count:
                phicount[0, phase] += 1
                for i in range(len(sequence)):set1_counts_5_exon[(phase+i)%3,sequence[i]] += 1

            sequence = str(genes[gene].seq[ss + 6:ss + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 6:].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_5_intron[s] += 1

        for ss in set1Threes:
            if ss==trueThrees[-1]:
                continue
            corresponding5 = trueFives[np.where(trueThrees == ss)[0][0] + 1]
            if ss < trueFive1 or corresponding5 > trueThree1:
                continue
            phase = phaseSeqs[gene][ss + 4] - 10
            #phase2 = phaseSeqs[gene][ss + sreEffect3_exon] - 10
            lastbase=ss+sreEffect3_exon if len(phaseSeqs[gene])>=(ss+sreEffect3_exon+1) else len(phaseSeqs[gene])-1
            #endphase = phaseSeqs[gene][lastbase] - 10
            sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            count=True
            if phase not in [0,1,2]:# or phase2 not in [0,1,2]:
                count=False
                phicount[1, 3] += 1
                '''for i in range(lastbase,ss+3,-1):
                    if phaseSeqs[gene][i] >= 10:
                        sequence = str(genes[gene].seq[ss+4:i+1].lower())
                        phicount[1, phase] += 1
                        count=True
                        break
                if not count:
                    phicount[1, 3] += 1'''
                    #print(phase,phase2)
                    #print(gene,trueFive1,trueThree1,ss,corresponding5,phaseSeqs[gene][ss+1:corresponding5])
            #sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            '''if len(sequence) == 0:
                sequence = str(genes[gene].seq[ss + 4:].lower())
                count=False
            if len(sequence) < kmer: continue'''
            if count:#len(sequence)==sreEffect3_exon-3 and endphase in [0,1,2]:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                #if count:
                phicount[1, phase] += 1
                for i in range(len(sequence)): set1_counts_3_exon[(i+phase)%3,sequence[i]] += 1

            sequence = str(genes[gene].seq[ss - sreEffect3_intron:ss - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set1_counts_3_intron[s] += 1

        for ss in set2Fives:
            if ss == predFives[0]:
                continue
            corresponding3 = predThrees[np.where(predFives == ss)[0][0] - 1]
            if corresponding3 < trueFive1 or ss > trueThree1:
                continue
            fpcount[0]+=1
            sequence = str(genes[gene].seq[ss - sreEffect5_exon:ss - 3].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 3].lower())
            if len(sequence) < kmer: continue
            if len(sequence) == sreEffect5_exon - 3:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                for s in sequence: set2_counts_5_exon[s] += 1

            sequence = str(genes[gene].seq[ss + 6:ss + sreEffect5_intron + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 6:].lower())
            if len(sequence) < kmer: continue

            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_5_intron[s] += 1

        for ss in set2Threes:
            if ss == predThrees[-1]:
                continue
            corresponding5 = predFives[np.where(predThrees == ss)[0][0] + 1]
            if ss < trueFive1 or corresponding5 > trueThree1:
                continue
            fpcount[1]+=1
            sequence = str(genes[gene].seq[ss + 4:ss + sreEffect3_exon + 1].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[ss + 4:].lower())
            if len(sequence) < kmer: continue
            if len(sequence)==sreEffect3_exon-3:
                sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
                for s in sequence: set2_counts_3_exon[s] += 1

            sequence = str(genes[gene].seq[ss - sreEffect3_intron:ss - 19].lower())
            if len(sequence) == 0: sequence = str(genes[gene].seq[:ss - 19].lower())
            if len(sequence) < kmer: continue
            sequence = np.array([hashSequence(sequence[i:i + kmer]) for i in range(len(sequence) - kmer + 1)])
            for s in sequence: set2_counts_3_intron[s] += 1

    return (set1_counts_5_intron, set1_counts_5_exon, set1_counts_3_intron, set1_counts_3_exon,
            set2_counts_5_intron, set2_counts_5_exon, set2_counts_3_intron, set2_counts_3_exon, phicount,fpcount)


def get_hexamer_real_decoy_scores(geneNames, trueSeqs, decoySS, genes, kmer, sreEffect5_exon, sreEffect5_intron, sreEffect3_exon, sreEffect3_intron,start_stop):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon, true_counts_open, true_counts_close, decoy_counts_open, decoy_counts_close = get_hexamer_real_decoy_counts(geneNames, trueSeqs, decoySS, genes, kmer = kmer, sreEffect5_exon = sreEffect5_exon, sreEffect5_intron = sreEffect5_intron, sreEffect3_exon = sreEffect3_exon, sreEffect3_intron = sreEffect3_intron,start_stop=start_stop)
    
    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1

    true_counts_open = true_counts_open + 1
    true_counts_close = true_counts_close + 1
    decoy_counts_open = decoy_counts_open + 1
    decoy_counts_close = decoy_counts_close + 1

    true_counts_all=true_counts_open+true_counts_close
    decoy_counts_all=decoy_counts_open+decoy_counts_close

    true_frac_open=true_counts_open/true_counts_all
    true_frac_close=true_counts_close/true_counts_all
    decoy_frac_open=decoy_counts_open/decoy_counts_all
    decoy_frac_close=decoy_counts_close/decoy_counts_all
    
    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon

    score_open = true_frac_open / decoy_frac_open
    score_close = decoy_frac_close / true_frac_close

    '''score_open = true_counts_open/decoy_counts_open
    score_close = decoy_counts_close/true_counts_close'''
    
    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))) 
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    decoyFreqs_exon = np.exp(np.log(decoy_counts_exon) - np.log(np.sum(true_counts_exon)))
    
    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)) 
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)) 
                            - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    
    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron)) 
                                - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon)) 
                              - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    
    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron)) 
                                - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon)) 
                              - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))


    
    return(sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon,score_open,score_close)

def get_hexamer_real_decoy_scores_C(geneNames, trueSeqs, decoySS, genes, phaseSeqs, kmer, sreEffect5_exon,
                                  sreEffect5_intron, sreEffect3_exon, sreEffect3_intron, start_stop):
    # Get the real versus decoy scores for all hexamers
    true_counts_5_intron, true_counts_5_exon, true_counts_3_intron, true_counts_3_exon, decoy_counts_5_intron, decoy_counts_5_exon, decoy_counts_3_intron, decoy_counts_3_exon, true_counts_5_exon_phase, true_counts_3_exon_phase, phi, phi5 = get_hexamer_real_decoy_counts_C(
        geneNames, trueSeqs, decoySS, genes, phaseSeqs, kmer=kmer, sreEffect5_exon=sreEffect5_exon,
        sreEffect5_intron=sreEffect5_intron, sreEffect3_exon=sreEffect3_exon, sreEffect3_intron=sreEffect3_intron,
        start_stop=start_stop)

    # Add pseudocounts
    true_counts_5_intron = true_counts_5_intron + 1
    true_counts_5_exon = true_counts_5_exon + 1
    true_counts_3_intron = true_counts_3_intron + 1
    true_counts_3_exon = true_counts_3_exon + 1
    decoy_counts_5_intron = decoy_counts_5_intron + 1
    decoy_counts_5_exon = decoy_counts_5_exon + 1
    decoy_counts_3_intron = decoy_counts_3_intron + 1
    decoy_counts_3_exon = decoy_counts_3_exon + 1
    true_counts_5_exon_phase = (true_counts_5_exon_phase + 1)
    true_counts_3_exon_phase = (true_counts_3_exon_phase + 1)
    true_counts_exon_phase = (true_counts_5_exon_phase + true_counts_3_exon_phase)

    '''true_counts_open = true_counts_open + 1
    true_counts_close = true_counts_close + 1
    decoy_counts_open = decoy_counts_open + 1
    decoy_counts_close = decoy_counts_close + 1

    true_counts_all=true_counts_open+true_counts_close
    decoy_counts_all=decoy_counts_open+decoy_counts_close

    true_frac_open=true_counts_open/true_counts_all
    true_frac_close=true_counts_close/true_counts_all
    decoy_frac_open=decoy_counts_open/decoy_counts_all
    decoy_frac_close=decoy_counts_close/decoy_counts_all'''

    true_counts_intron = true_counts_5_intron + true_counts_3_intron
    true_counts_exon = true_counts_5_exon + true_counts_3_exon
    decoy_counts_intron = decoy_counts_5_intron + decoy_counts_3_intron
    decoy_counts_exon = decoy_counts_5_exon + decoy_counts_3_exon

    '''score_open = true_frac_open/decoy_frac_open
    score_close = decoy_frac_close/true_frac_close'''

    trueFreqs_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron)))
    decoyFreqs_intron = np.exp(np.log(decoy_counts_intron) - np.log(np.sum(decoy_counts_intron)))
    #trueFreqs_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon)))
    #decoyFreqs_exon3 = np.exp(np.log(decoy_counts_3_exon) - np.log(np.sum(true_counts_3_exon)))
    #decoyFreqs_exon5 = np.exp(np.log(decoy_counts_5_exon) - np.log(np.sum(true_counts_5_exon)))

    sreScores_intron = np.exp(np.log(true_counts_intron) - np.log(np.sum(true_counts_intron))
                              - np.log(decoy_counts_intron) + np.log(np.sum(decoy_counts_intron)))
    #sreScores_exon = np.exp(np.log(true_counts_exon) - np.log(np.sum(true_counts_exon))
    #                        - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))

    sreScores3_intron = np.exp(np.log(true_counts_3_intron) - np.log(np.sum(true_counts_3_intron))
                               - np.log(decoy_counts_3_intron) + np.log(np.sum(decoy_counts_3_intron)))
    #sreScores3_exon = np.exp(np.log(true_counts_3_exon) - np.log(np.sum(true_counts_3_exon))
    #                          - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))

    sreScores5_intron = np.exp(np.log(true_counts_5_intron) - np.log(np.sum(true_counts_5_intron))
                               - np.log(decoy_counts_5_intron) + np.log(np.sum(decoy_counts_5_intron)))
    #sreScores5_exon = np.exp(np.log(true_counts_5_exon) - np.log(np.sum(true_counts_5_exon))
    #                          - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))

    sreScore3_exon_p1 = np.exp(np.log(true_counts_3_exon_phase[0]) - np.log(np.sum(true_counts_3_exon_phase[0]))
                               - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    sreScore3_exon_p2 = np.exp(np.log(true_counts_3_exon_phase[1]) - np.log(np.sum(true_counts_3_exon_phase[1]))
                               - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    sreScore3_exon_p3 = np.exp(np.log(true_counts_3_exon_phase[2]) - np.log(np.sum(true_counts_3_exon_phase[2]))
                               - np.log(decoy_counts_3_exon) + np.log(np.sum(decoy_counts_3_exon)))
    sreScores3_exon = np.array([sreScore3_exon_p1, sreScore3_exon_p2, sreScore3_exon_p3])

    sreScore5_exon_p1 = np.exp(np.log(true_counts_5_exon_phase[0]) - np.log(np.sum(true_counts_5_exon_phase[0]))
                               - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    sreScore5_exon_p2 = np.exp(np.log(true_counts_5_exon_phase[1]) - np.log(np.sum(true_counts_5_exon_phase[1]))
                               - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    sreScore5_exon_p3 = np.exp(np.log(true_counts_5_exon_phase[2]) - np.log(np.sum(true_counts_5_exon_phase[2]))
                               - np.log(decoy_counts_5_exon) + np.log(np.sum(decoy_counts_5_exon)))
    sreScores5_exon = np.array([sreScore5_exon_p1, sreScore5_exon_p2, sreScore5_exon_p3])

    sreScore_exon_p1 = np.exp(np.log(true_counts_exon_phase[0]) - np.log(np.sum(true_counts_exon_phase[0]))
                               - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    sreScore_exon_p2 = np.exp(np.log(true_counts_exon_phase[1]) - np.log(np.sum(true_counts_exon_phase[1]))
                               - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    sreScore_exon_p3 = np.exp(np.log(true_counts_exon_phase[2]) - np.log(np.sum(true_counts_exon_phase[2]))
                               - np.log(decoy_counts_exon) + np.log(np.sum(decoy_counts_exon)))
    sreScores_exon = np.array([sreScore_exon_p1, sreScore_exon_p2, sreScore_exon_p3])

    ###
    '''sreScore3_exon_f1 = np.exp(np.log(true_counts_3_exon_phase[0]) - np.log(np.sum(true_counts_3_exon_phase[0])))
    sreScore3_exon_f2 = np.exp(np.log(true_counts_3_exon_phase[1]) - np.log(np.sum(true_counts_3_exon_phase[1])))
    sreScore3_exon_f3 = np.exp(np.log(true_counts_3_exon_phase[2]) - np.log(np.sum(true_counts_3_exon_phase[2])))
    sreScore3_exon_f = np.array([sreScore3_exon_f1, sreScore3_exon_f2, sreScore3_exon_f3],dtype=np.float64)

    sreScore5_exon_f1 = np.exp(np.log(true_counts_5_exon_phase[0]) - np.log(np.sum(true_counts_5_exon_phase[0])))
    sreScore5_exon_f2 = np.exp(np.log(true_counts_5_exon_phase[1]) - np.log(np.sum(true_counts_5_exon_phase[1])))
    sreScore5_exon_f3 = np.exp(np.log(true_counts_5_exon_phase[2]) - np.log(np.sum(true_counts_5_exon_phase[2])))
    sreScore5_exon_f = np.array([sreScore5_exon_f1, sreScore5_exon_f2, sreScore5_exon_f3],dtype=np.float64)'''

    return (sreScores_intron, sreScores_exon, sreScores3_intron, sreScores3_exon, sreScores5_intron, sreScores5_exon, phi, phi5)

def score_sequences(sequences, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k = 6, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    batch_size = len(sequences)
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
        
    return np.exp(emissions5.base), np.exp(emissions3.base)

def score_sequences_C(sequences, double [:] phi, double [:, :, :] exonicSREs5s, double [:, :, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k = 6, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    batch_size = len(sequences)
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))

    # Collect the lengths of each sequence in the batch
    for g in range(batch_size):
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)

    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))

    cdef double ESREc3, ESREc5, F1phi, F2phi, F3phi, Gi

    # Get the emissions and apply sre scores to them
    for g in range(batch_size):
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        for i in range(80, lengths[g] - ssRange):  #position of 5'SS
            #print('shape:',exonicSREs5s.shape,'p1g1:',exonicSREs5s[0,g,0])
            #print(exonicSREs5s[0,g,i-81+3*0])
            F1phi = np.exp(np.sum([exonicSREs5s.base[0, g, i - 80 + 3 * c] for c in range(24)])) * phi[0]
            F2phi = np.exp(np.sum([exonicSREs5s.base[1, g, i - 80 + 3 * c] for c in range(24)])) * phi[1]
            F3phi = np.exp(np.sum([exonicSREs5s.base[2, g, i - 80 + 3 * c] for c in range(24)])) * phi[2]
            #Gi = np.exp(np.sum([decoySRE5s.base[g, i - 81 + c] for c in range(80 - k + 1)]))
            #print(decoySREs.shape,decoySREs.base[g,i-81])
            #print(np.sum([decoySREs.base[g,i-81+c] for c in range(80-k+1)]),F1phi+F2phi+F3phi)
            ESREc5 = np.log2((F1phi + F2phi + F3phi))# / (Gi))
            emissions5.base[g, i + ssRange] += ESREc5
        #emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        #emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]

        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]

        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])

        # 3'SS exonic effects (downstream)
        ssRange = 3
        for i in range(lengths[g] - 83):
            F1phi = np.exp(np.sum([exonicSREs3s.base[0, g, i + 1 + 3 * c] for c in range(25)])) * phi[0]
            F2phi = np.exp(np.sum([exonicSREs3s.base[1, g, i + 1 + 3 * c] for c in range(25)])) * phi[1]
            F3phi = np.exp(np.sum([exonicSREs3s.base[2, g, i + 1 + 3 * c] for c in range(25)])) * phi[2]
            #Gi = np.exp(np.sum(decoySRE3s.base[g, i + 1:i + 1 + 80 - k + 1]))
            ESREc3 = np.log2((F1phi + F2phi + F3phi))#/ (Gi))
            emissions3.base[g, i - ssRange] += ESREc3
        '''emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])'''

    return np.exp(emissions5.base), np.exp(emissions3.base)

def cass_accuracy_metrics(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, B3 = 3, B5 = 5):
    # Get the best cutoff and the associated metrics for the CASS scored sequences
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0, min_score
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    best_f1 = 0
    best_cutoff = 0
    for i, cutoff in enumerate(all_scores):
        if all_scores_bool[i] == 0: continue
        true_positives = np.sum(all_scores_bool[i:])
        false_negatives = num_all_positives - true_positives
        false_positives = num_all - i - true_positives
        
        ssSens = true_positives / (true_positives + false_negatives)
        ssPrec = true_positives / (true_positives + false_positives)
        f1 = 2 / (1/ssSens + 1/ssPrec)
        if f1 >= best_f1:
            best_f1 = f1
            best_cutoff = cutoff
            best_sens = ssSens
            best_prec = ssPrec
        
    return best_sens, best_prec, best_f1, best_cutoff
    
def cass_accuracy_metrics_set_cutoff(scored_sequences_5, scored_sequences_3, geneNames, trueSeqs, cutoff, B3 = 3, B5 = 5):
    # Get the associated metrics for the CASS scored sequences with a given cutoff
    true_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] == B5]): true_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] == B3]): true_scores.append(score)
    min_score = np.min(true_scores)
    if np.isnan(min_score): 
        return 0, 0, 0
    
    all_scores = []
    for g, gene in enumerate(geneNames):
        L = len(trueSeqs[gene])
        for score in np.log2(scored_sequences_5[g,:L][trueSeqs[gene] != B5]): 
            if score > min_score: all_scores.append(score)
        for score in np.log2(scored_sequences_3[g,:L][trueSeqs[gene] != B3]): 
            if score > min_score: all_scores.append(score)
    
    all_scores = np.array(true_scores + all_scores)
    all_scores_bool = np.zeros(len(all_scores), dtype=np.dtype("i"))
    all_scores_bool[:len(true_scores)] = 1
    sort_inds = np.argsort(all_scores)
    all_scores = all_scores[sort_inds]
    all_scores_bool = all_scores_bool[sort_inds]
    
    num_all_positives = len(true_scores)
    num_all = len(all_scores)
    
    true_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 1))
    false_negatives = num_all_positives - true_positives
    false_positives = np.sum((all_scores > cutoff)&(all_scores_bool == 0))
    
    ssSens = true_positives / (true_positives + false_negatives)
    ssPrec = true_positives / (true_positives + false_positives)
    f1 = 2 / (1/ssSens + 1/ssPrec)
        
    return ssSens, ssPrec, f1

def order_genes(geneNames, num_threads, genes):
    # Re-order genes to feed into parallelized prediction algorithm to use parallelization efficiently
    # geneNames: list of names of genes to re-order based on length 
    # num_threads: number of threads available to parallelize across
    lengthsOfGenes = np.array([len(str(genes[gene].seq)) for gene in geneNames])
    geneNames = geneNames[np.argsort(lengthsOfGenes)]
    geneNames = np.flip(geneNames)

    # ordering the genes for optimal processing
    l = len(geneNames)
    ind = l - l//num_threads
    longest_thread = []
    for i in np.flip(range(num_threads)):
        longest_thread.append(ind)
        ind -= (l//num_threads + int(i<=(l%num_threads)))
    
    indices = longest_thread.copy()
    for i in range(1,l//num_threads):
        indices += list(np.array(longest_thread) + i)
    
    ind = l//num_threads
    for i in range(l%num_threads): indices.append(ind + i*l%num_threads)

    indices = np.argsort(indices)
    return(geneNames[indices])


def viterbi(sequences, transitions, long [:,:] coordinates_fl, double [:] pIL, double [:] pELS, double [:] pELF, double [:] pELM, double [:] pELL, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, double rfOpen, double rfClose, long [:,:] start_stop, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, int lmode=False, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pME, p1E, pEE, pEO
    
    batch_size = len(sequences)
    
#     cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
     
    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)


    cdef int[:, :] stopcodons1
    cdef int[:, :] stopcodons2
    cdef int[:, :] stopcodons3
    cdef int maxlen
    maxlen = np.max([len(x) for x in sequences])
    stopcodons1=np.zeros((batch_size,maxlen),dtype=np.int32)
    stopcodons2 = np.zeros((batch_size, maxlen), dtype=np.int32)
    stopcodons3 = np.zeros((batch_size, maxlen), dtype=np.int32)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size):
        rfStopCodon = stopCodonCount(sequences[g])
        for p in range(rfStopCodon.shape[1]):
            stopcodons1[g][p] = rfStopCodon[0][p]
            stopcodons2[g][p] = rfStopCodon[1][p]
            stopcodons3[g][p] = rfStopCodon[2][p]
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
    
    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))
    
    # Initialize the first and single exon probabilities
    cdef double [:] ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E

    cdef int closeFrameCount1,closeFrameCount2,closeFrameCount3,allclose, rfo, rfc
    cdef double result
    cdef int diff

    #cdef int[:] rfStopCodon


    #stopcodons_array = np.array(stopcodons)

    # Convert NumPy array to Cython memoryview
    #cdef cnp.ndarray[np.int32_t, ndim=2] stopcodons_view = stopcodons_array.view(dtype=np.int32).reshape(stopcodons_array.shape)
    cdef double openscore=0#np.log(rfOpen)
    cdef double closescore=np.log(rfClose)
    print(closescore)

    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        #seq = seqs[g]
        #slen = strlen(seq)
        #rfStopCodon = stopcodons[g]
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]

            for d in range(t, 0, -1):
                closeFrameCount1 = stopcodons1[g,t-3+1]-stopcodons1[g,t-d-1]
                closeFrameCount2 = stopcodons2[g, t-3+1] - stopcodons2[g, t - d - 1]
                closeFrameCount3 = stopcodons3[g, t-3+1] - stopcodons3[g, t - d - 1]

                allclose=(closeFrameCount1 > 0)+(closeFrameCount2 > 0)+(closeFrameCount3 > 0)

                # Calculate rfo and rfc as boolean flags
                rfo = (allclose <3)
                rfc = (allclose ==3)

                # Calculate the result in nogil context
                '''if ((t<lengths[g]-1 and d<=500) or t==lengths[g]-1) and lmode:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1]) + (rfo * rfOpen[g] - rfc * rfClose[g])*(1-(t==d))*(1-(t==(lengths[g]-1)))

                    if result > Five[g, t]:
                        traceback5[g, t] = d
                        Five[g, t] = result
                if lmode==False:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1]) + (
                                rfo * rfOpen[g] - rfc * rfClose[g])*(1-(t==d))*(1-(t==(lengths[g]-1)))

                    if result > Five[g, t]:
                        traceback5[g, t] = d
                        Five[g, t] = result'''
                if (t)<start_stop[g][1] and (t-d-1)>start_stop[g][0]:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1]) - ( rfc * closescore)# * (1 - (t == d)) * (1 - (t == (lengths[g] - 1)))
                else:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1])

                if result > Five[g, t]:
                    traceback5[g, t] = d
                    Five[g, t] = result
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i
                
        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
        else:
            loglik[g] = ES[g]
        
    return bestPath.base, loglik.base, emissions5.base, emissions3.base
from libc.math cimport log
def viterbi_C(sequences, transitions, long [:,:] coordinates_fl, double [:] pIL, double [:] pELS, double [:] pELF, double [:] pELM, double [:] pELL, double [:] phi, double [:] phi5, double [:, :, :] exonicSREs5s, double [:, :, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, long [:,:] start_stop, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80,   meDir = '', double rfClose=1.0): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pME, p1E, pEE, pEO

    batch_size = len(sequences)

#     cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))

    # Collect the lengths of each sequence in the batch
    for g in range(batch_size):
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)

    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))

    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))

    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1

    # Convert inputs to log space
    transitions = np.log(transitions)
    pIL = np.log(pIL)
    pELS = np.log(pELS)
    pELF = np.log(pELF)
    pELM = np.log(pELM)
    pELL = np.log(pELL)

    cdef int[:, :] stopcodons1
    cdef int[:, :] stopcodons2
    cdef int[:, :] stopcodons3
    cdef int maxlen
    maxlen = np.max([len(x) for x in sequences])
    stopcodons1=np.zeros((batch_size,maxlen),dtype=np.int32)
    stopcodons2 = np.zeros((batch_size, maxlen), dtype=np.int32)
    stopcodons3 = np.zeros((batch_size, maxlen), dtype=np.int32)

    cdef double ESREc3, ESREc5, F1phi, F2phi, F3phi
    cdef double [:,:,:] Fl=np.zeros((batch_size,3,L), dtype=np.dtype("d"))
    cdef double [:,:,:] Fl3=np.zeros((batch_size,3,L), dtype=np.dtype("d"))

    #cdef double F1phi_prestop, F2phi_prestop, F3phi_prestop
    cdef double [:,:,:] Fl_prestop=np.zeros((batch_size,3,L), dtype=np.dtype("d"))
    cdef double [:,:,:] Fl3_prestop=np.zeros((batch_size,3,L), dtype=np.dtype("d"))
    cdef int temp5,temp3

    #cdef double [:,:] e51,e52,e53
    #e51 = exonicSREs5s
    # Get the emissions and apply sre scores to them
    # change overall emissions to ESREc
    cdef int numkmer = int((sreEffect5_exon-k+1)/3)

    for g in range(batch_size):
        # 5'SS exonic effects (upstream)
        #print(g)
        rfStopCodon = stopCodonCount(sequences[g])
        for p in range(rfStopCodon.shape[1]):
            stopcodons1[g][p] = rfStopCodon[0][p]
            stopcodons2[g][p] = rfStopCodon[1][p]
            stopcodons3[g][p] = rfStopCodon[2][p]
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        #sequence = np.array([hashSequence(sequences[g][i:i + k]) for i in range(len(sequences[g]) - k + 1)])
        for i in range(lengths[g]):#position of 5'SS

            F1phi=np.exp(np.sum(np.array([exonicSREs5s.base[0,g,i-sreEffect5_exon+3*c] for c in range(numkmer) if i-sreEffect5_exon+3*c>=0])))#+[np.log(phi5[0])])))#*phi5[0]
            F2phi=np.exp(np.sum(np.array([exonicSREs5s.base[1,g,i-sreEffect5_exon+3*c] for c in range(numkmer) if i-sreEffect5_exon+3*c>=0])))#+[np.log(phi5[1])])))#*phi5[1]
            F3phi=np.exp(np.sum(np.array([exonicSREs5s.base[2,g,i-sreEffect5_exon+3*c] for c in range(numkmer) if i-sreEffect5_exon+3*c>=0])))#+[np.log(phi5[2])])))#*phi5[2]
            Fl[g,0,i]+=F1phi
            Fl[g,1,i]+=F2phi
            Fl[g,2,i]+=F3phi

            '''_, temp5 = first_last_stop(sequences[g][i - sreEffect5_exon:i - ssRange])
            if temp5 != -1:
                if temp5 < 80 - 6 - 3:
                    F1phi = np.exp(np.sum(np.array([exonicSREs5s.base[0, g, i - sreEffect5_exon + 3 * c] for c in
                                                    range(int(temp5 / 3) + 1, numkmer) if
                                                    i - sreEffect5_exon + 3 * c >= 0])))
                    F2phi = np.exp(np.sum(np.array([exonicSREs5s.base[1, g, i - sreEffect5_exon + 3 * c] for c in
                                                    range(int(temp5 / 3) + 1, numkmer) if
                                                    i - sreEffect5_exon + 3 * c >= 0])))
                    F3phi = np.exp(np.sum(np.array([exonicSREs5s.base[2, g, i - sreEffect5_exon + 3 * c] for c in
                                                    range(int(temp5 / 3) + 1, numkmer) if
                                                    i - sreEffect5_exon + 3 * c >= 0])))
                else:
                    F1phi = 1 / 3
                    F2phi = 1 / 3
                    F3phi = 1 / 3
                Fl_prestop[g, 0, i] += F1phi
                Fl_prestop[g, 1, i] += F2phi
                Fl_prestop[g, 2, i] += F3phi
            else:
                Fl_prestop[g, 0, i] += F1phi
                Fl_prestop[g, 1, i] += F2phi
                Fl_prestop[g, 2, i] += F3phi  '''



        #emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        #emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]

        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]


        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])


        # 3'SS exonic effects (downstream)
        ssRange = 3
        #emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        for i in range(lengths[g]):
            F1phi=np.exp(np.sum(np.array([exonicSREs3s.base[0,g,i+1+3*c] for c in range(1,numkmer+1) if i+1+3*c<lengths[g]-1-k])))#+[np.log(phi[0])])))#*phi[0]
            F2phi = np.exp(np.sum(np.array([exonicSREs3s.base[1,g,i+1+3*c] for c in range(1,numkmer+1) if i+1+3*c<lengths[g]-1-k])))#+[np.log(phi[1])])))#* phi[1]
            F3phi = np.exp(np.sum(np.array([exonicSREs3s.base[2,g,i+1+3*c] for c in range(1,numkmer+1) if i+1+3*c<lengths[g]-1-k])))#+[np.log(phi[2])])))#* phi[2]
            Fl3[g, 0, i] += F1phi
            Fl3[g, 1, i] += F2phi
            Fl3[g, 2, i] += F3phi

            '''temp3, _ = first_last_stop(sequences[g][i + 4:i + sreEffect3_exon])
            if temp3 != -1:
                if temp3 > 9:
                    F1phi = np.exp(np.sum(np.array(
                        [exonicSREs3s.base[0, g, i + 1 + 3 * c] for c in range(1, int(temp3 / 3) + 1) if
                         i + 1 + 3 * c < lengths[g] - 1 - k])))
                    F2phi = np.exp(np.sum(np.array(
                        [exonicSREs3s.base[1, g, i + 1 + 3 * c] for c in range(1, int(temp3 / 3) + 1) if
                         i + 1 + 3 * c < lengths[g] - 1 - k])))
                    F3phi = np.exp(np.sum(np.array(
                        [exonicSREs3s.base[2, g, i + 1 + 3 * c] for c in range(1, int(temp3 / 3) + 1) if
                         i + 1 + 3 * c < lengths[g] - 1 - k])))
                else:
                    F1phi = 1 / 3
                    F2phi = 1 / 3
                    F3phi = 1 / 3
                Fl3_prestop[g, 0, i] += F1phi
                Fl3_prestop[g, 1, i] += F2phi
                Fl3_prestop[g, 2, i] += F3phi
            else:
                Fl3_prestop[g, 0, i] += F1phi
                Fl3_prestop[g, 1, i] += F2phi
                Fl3_prestop[g, 2, i] += F3phi'''

            #Gi=np.exp(np.sum(decoySRE3s.base[g,i+1:i+1+80-k+1]))
            #ESREc3=np.log((F1phi*phi[0]+F2phi*phi[1]+F3phi*phi[2]))#/(Gi))/6
            #emissions3.base[g,i]+=ESREc3
        #emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] =exonicSREs3s.base[g,:lengths[g]-k+1]#+= np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        #emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])'''

    # Convert the transition vector into named probabilities
    pME = transitions[0]
    p1E = np.log(1 - np.exp(pME))
    pEE = transitions[1]
    pEO = np.log(1 - np.exp(pEE))



    # Initialize the first and single exon probabilities
    cdef double [:] ES = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): ES[g] = pELS[L-1] + p1E

    cdef int closeFrameCount1,closeFrameCount2,closeFrameCount3,allclose, rfo, rfc
    cdef double result
    cdef int diff
    #stopcodons_array = np.array(stopcodons)
    cdef int phase

    cdef double openscore = 0  #np.log(rfOpen)
    cdef double closescore = np.log(rfClose)

    cdef double maxv
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        #seq = seqs[g]
        #slen = strlen(seq)
        #rfStopCodon = stopcodons[g]
        #geneseq3 = new_sequences[g]
        for t in range(1,lengths[g]):
            Five[g,t] = pELF[t-1]
            #Three[g, t] += (emissions3[g, t])
            #Five[g, t] += (emissions5[g, t])
            for d in range(t, 0, -1):

                closeFrameCount1 = stopcodons1[g, t - 3 + 1] - stopcodons1[g, t - d - 1]
                closeFrameCount2 = stopcodons2[g, t - 3 + 1] - stopcodons2[g, t - d - 1]
                closeFrameCount3 = stopcodons3[g, t - 3 + 1] - stopcodons3[g, t - d - 1]

                allclose = (closeFrameCount1 > 0) + (closeFrameCount2 > 0) + (closeFrameCount3 > 0)

                # Calculate rfo and rfc as boolean flags
                rfo = (allclose < 3)
                rfc = (allclose == 3)
                # Calculate the result in nogil context
                # 3'ss = t-d-1; 5'ss=t

                #result = pEE + (Three[g, t - d - 1] + log(Fl3[g,0,t-d-1]*phi[0]+Fl3[g,1,t-d-1]*phi[1]+Fl3[g,2,t-d-1]*phi[2])) + pELM[d - 1] + log(Fl[g,0,t]*phi[phase] + Fl[g,1,t]*phi[(phase+1)%3] + Fl[g,2,t]*phi[(phase+2)%3])
                # + (rfo * rfOpen[g] - rfc * rfClose[g])
                #phase = (t - 83 - (t - d)) % 3
                #temp5 = ( log((Fl[g, phase, t] * phi[0] + Fl[g, (phase + 1) % 3, t] * phi[1] + Fl[g, (phase + 2) % 3, t] * phi[2]))) + emissions5[g,t]
                #temp3 = (log((Fl3[g,0,t-d-1])*phi[0]+Fl3[g,1,t-d-1]*phi[1]+Fl3[g,2,t-d-1]*phi[2])*3) +emissions3[g,t-d-1]
                if (t)>coordinates_fl[g][0] and (t-d-1)<coordinates_fl[g][1]:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1]) - ( rfc * closescore)# * (1 - (t == d)) * (1 - (t == (lengths[g] - 1)))
                else:
                    result = (pEE + Three[g, t - d - 1] + pELM[d - 1])
                if result > Five[g, t]:
                    traceback5[g, t] = d
                    Five[g, t] = result

                # 3'SS
                #phase = (t - 83 - (t - traceback5[g,t-d-1])) % 3
                #temp5 = (log((Fl[g, phase, t] * phi[0] + Fl[g, (phase + 1) % 3, t] * phi[1] + Fl[g, (phase + 2) % 3, t] * phi[2]))*3) + emissions5[g,t-d-1]
                #temp3 = (log((Fl3[g, 0, t]* phi[0] + Fl3[g, 1, t] * phi[1] + Fl3[g, 2, t] *phi[2])*3) + emissions3[g,t])
                if Five[g,t-d-1] + pIL[d-1]  > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]


            Three[g, t] += (emissions3[g, t])# + ESREc3)
            Five[g, t] += (emissions5[g, t])
            '''maxv=log(Fl[g,0,t]*3)
            if log(Fl[g,1,t]*3)>maxv: maxv=log(Fl[g,1,t]*3)
            if log(Fl[g,2,t]*3)>maxv:maxv=log(Fl[g,2,t]*3)
            Five[g,t]+=maxv
            maxv = log(Fl3[g, 0, t]*3)
            if log(Fl3[g, 1, t]*3) > maxv: maxv = log(Fl3[g, 1, t]*3)
            if log(Fl3[g, 2, t]*3) > maxv: maxv = log(Fl3[g, 2, t]*3)
            Three[g, t] += maxv'''
            '''if traceback5[g,t]!=0:
                if traceback5[g,t]>=sreEffect5_exon:
                    phase = (- sreEffect5_exon +traceback5[g,t]) % 3
                else:
                    phase = (3-(sreEffect5_exon - traceback5[g, t]) % 3)%3
                if traceback5[g,t]!=t:
                    Five[g, t] +=  log((Fl[g, phase, t] * phi[0] + Fl[g, (phase + 1) % 3, t] * phi[1] + Fl[g, (phase + 2) % 3, t] * phi[2])*3)
            else:
                Five[g, t] += log((Fl[g, 0, t] * phi5[0] + Fl[g, 1, t] * phi5[1] + Fl[g, 2, t] * phi5[2])*3)
            #if traceback3[g,t]!=0:
            Three[g,t]+=log((Fl3[g,0,t]*phi[0]+Fl3[g,1,t]*phi[1]+Fl3[g,2,t]*phi[2])*3)'''

            '''if t<start_stop[g][1]:
                phase = (- sreEffect5_exon + traceback5[g, t]-1) % 3
                Five[g, t] += log((Fl[g, 0, t] * phi[phase] + Fl[g, 1, t] * phi[(1+phase)%3] + Fl[g, 2, t] * phi[(2+phase)%3]) )
            else:'''
            Five[g, t] += log((Fl[g, 0, t] * phi5[0] + Fl[g, 1, t] * phi5[1] + Fl[g, 2, t] * phi5[2]) )
            #if t>start_stop[g][0]:
            Three[g, t] += log((Fl3[g, 0, t] * phi[0] + Fl3[g, 1, t] * phi[1] + Fl3[g, 2, t] * phi[2]) )
            '''else:
                Three[g, t] += log((Fl3[g, 0, t] * (1/3) + Fl3[g, 1, t] * (1/3) + Fl3[g, 2, t] * (1/3)) )'''

            #3'ss is t-traceback5[g,t]-1
            #
        #go back through traceback3 to find the last 3'SS?
        '''for i in range(lengths[g]-1,-1,-1):
            if traceback3[g,i]!=0:
                Three[g,i]=Three[g,i]-log((Fl3[g,0,i]*phi[0]+Fl3[g,1,i]*phi[1]+Fl3[g,2,i]*phi[2])*3)#+log((Fl3_prestop[g,0,i]*phi[0]+Fl3_prestop[g,1,i]*phi[1]+Fl3_prestop[g,2,i]*phi[2])*3)
                break'''
        # TODO: Add back in single exon case for if we ever come back to that
        for i in range(1, lengths[g]):
            if pME + Three[g,i] + pEO + pELL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = pME + Three[g,i] + pEO + pELL[lengths[g]-i-2]
                tbindex[g] = i

        if ES[g] <= loglik[g]: # If the single exon case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1
        else:
            loglik[g] = ES[g]

    return bestPath.base, loglik.base, emissions5.base, emissions3.base

def viterbi_intron(sequences, double pIO, double [:] pIL, double [:] pELM, double [:, :] exonicSREs5s, double [:, :] exonicSREs3s, double [:, :] intronicSREs5s, double [:, :] intronicSREs3s, int k, int sreEffect5_exon = 80, int sreEffect5_intron = 80, int sreEffect3_exon = 80, int sreEffect3_intron = 80, meDir = ''): #/home/kmccue/projects/scfg/code/maxEnt/triplets-9mer-max/
    # Inputs:
    #  - sequences: batch_size lenght list of sequence strings, maybe need about 21 kb/base for the longest string
    #  - transitions: list of two values [pME, pEE]
    #  - pIL: numpy array of intron length distribution
    #  - pELS: decay coefficient for transcribing state eg 1/10000
    #  - pELF: numpy array of first exon length distribution
    #  - pELM: numpy array of middle exon length distribution
    #  - pELL: numpy array of last exon length distribution
    #  - delay: number of bases to delay predictions by
    #  - sreScores: sre scores for each kmer
    #  - meDir: location of the maxent model lookups
    # Outputs:
    #  - bestPath: numpy array containing the cotranscriptional viterbi best paths for each input sequence
    #  - loglik: numpy array containing the calculated log likelihoods
    #  - tbStates: numpy array containing the states at which a traceback event occurred
    #  - tblast3: numpy array containing the last 3'ss when which a traceback event occurred
    #  - emissions5: final 5' emissions
    #  - emissions3: final 3' emissions
    #  - traceback5: the 5'ss tracebacks
    #  - traceback3: the 3'ss tracebacks
    cdef int batch_size, g, L, d, i, t, ssRange
    cdef double pEE
    
    batch_size = len(sequences)
    
#     cdef int [:] t = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] tbindex = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef int [:] lengths = np.zeros(batch_size, dtype=np.dtype("i"))
    cdef double [:] loglik = np.log(np.zeros(batch_size, dtype=np.dtype("d")))
    
    # Collect the lengths of each sequence in the batch
    for g in range(batch_size): 
        lengths[g] = len(sequences[g])
    L = np.max(lengths.base)
    
    cdef double [:, :] emissions3 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef double [:, :] emissions5 = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
     
    cdef double [:, :] Three = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))    
    cdef double [:, :] Five = np.log(np.zeros((batch_size, L), dtype=np.dtype("d")))
    cdef int [:, :] traceback5 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] traceback3 = np.zeros((batch_size, L), dtype=np.dtype("i")) + L
    cdef int [:, :] bestPath = np.zeros((batch_size, L), dtype=np.dtype("i"))
    
    # Rewind state vars
    cdef int exon = 2
    cdef int intron = 1
     
    # Convert inputs to log space
    pIO = np.log(pIO)
    pEE = np.log(1 - np.exp(pIO))
    pIL = np.log(pIL)
    pELM = np.log(pELM)
    
    # Get the emissions and apply sre scores to them
    for g in range(batch_size): 
        # 5'SS exonic effects (upstream)
        ssRange = 3
        emissions5.base[g,:lengths[g]] = np.log(maxEnt5_single(sequences[g].lower(), meDir))
        emissions5.base[g,k+ssRange:lengths[g]] += np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions5.base[g,sreEffect5_exon+1:lengths[g]] -= np.cumsum(exonicSREs5s.base[g,:lengths[g]-k+1])[:-(sreEffect5_exon+1)+(k-1)]
        
        # 3'SS intronic effects (upstream)
        ssRange = 19
        emissions3.base[g,:lengths[g]] = np.log(maxEnt3_single(sequences[g].lower(), meDir))
        emissions3.base[g,k+ssRange:lengths[g]] += np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-1-ssRange]
        emissions3.base[g,sreEffect3_intron+1:lengths[g]] -= np.cumsum(intronicSREs3s.base[g,:lengths[g]-k+1])[:-(sreEffect3_intron+1)+(k-1)]
        
        # 5'SS intronic effects (downstream)
        ssRange = 4
        emissions5.base[g,:lengths[g]-sreEffect5_intron] += np.cumsum(intronicSREs5s.base[g,:lengths[g]-k+1])[sreEffect5_intron-k+1:]
        emissions5.base[g,lengths[g]-sreEffect5_intron:lengths[g]-k+1-ssRange] += np.sum(intronicSREs5s.base[g,:lengths[g]-k+1])
        emissions5.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(intronicSREs5s.base[g,ssRange:lengths[g]-k+1])
        
        # 3'SS exonic effects (downstream)
        ssRange = 3
        emissions3.base[g,:lengths[g]-sreEffect5_exon] += np.cumsum(exonicSREs3s.base[g,:lengths[g]-k+1])[sreEffect5_exon-k+1:]
        emissions3.base[g,lengths[g]-sreEffect5_exon:lengths[g]-k+1-ssRange] += np.sum(exonicSREs3s.base[g,:lengths[g]-k+1])
        emissions3.base[g,:lengths[g]-k+1-ssRange] -= np.cumsum(exonicSREs3s.base[g,ssRange:lengths[g]-k+1])
    
    # Initialize the first and single exon probabilities
    cdef double [:] IS = np.zeros(batch_size, dtype=np.dtype("d"))
    for g in range(batch_size): IS[g] = pIL[L-1] + pIO
    
    for g in prange(batch_size, nogil=True): # parallelize over the sequences in the batch
        for t in range(1,lengths[g]):
            Three[g,t] = pIL[t-1] + pEE
            
            for d in range(t,0,-1):
                # 5'SS
                if Three[g,t-d-1] + pELM[d-1] > Five[g,t]:
                    traceback5[g,t] = d
                    Five[g,t] = Three[g,t-d-1] + pELM[d-1]
            
                # 3'SS
                if Five[g,t-d-1] + pIL[d-1] > Three[g,t]:
                    traceback3[g,t] = d
                    Three[g,t] = Five[g,t-d-1] + pIL[d-1]
                    
            Five[g,t] += emissions5[g,t]
            Three[g,t] += emissions3[g,t]
            
        for i in range(1, lengths[g]):
            if Five[g,i] + pIO + pIL[lengths[g]-i-2] > loglik[g]:
                loglik[g] = Five[g,i] + pIO + pIL[lengths[g]-i-2]
                tbindex[g] = i
                
        if IS[g] <= loglik[g]: # If the single intron case isn't better, trace back
            while 0 < tbindex[g]:
                bestPath[g,tbindex[g]] = 5
                tbindex[g] -= traceback5[g,tbindex[g]] + 1 
                bestPath[g,tbindex[g]] = 3
                tbindex[g] -= traceback3[g,tbindex[g]] + 1
        else:
            loglik[g] = IS[g]
        
    return bestPath.base, loglik.base, emissions5.base, emissions3.base


