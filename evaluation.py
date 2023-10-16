import numpy as np
from scipy.spatial import distance_matrix
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
import inspect

'''
Author: Long Chen
Support:
    - higher dimensional data
    - evaluation in 'area' mode and 'curve' mode
    - input as label map or stacked binary maps
    - matrics: 
        - averagePrecision, aggregatedPricision
        - averageRecall, aggregatedRecall
        - averageF1, aggregatedF1
        - aggregatedJaccard, instanceAveragedJaccard
        - aggregatedDice, instanceaverageDice
        - SBD (symmetric best Dice)
'''

def map2stack(M, bg_label=0):
    '''
    Args:
        M: H x W x (1)
        bg_label: label of the background
    Return:
        S: C x H x W
    '''
    M = np.squeeze(M)
    labels = np.unique(M[M!=bg_label])
    S = np.ones((len(labels), M.shape[0], M.shape[1]), bool)
    for idx, l in enumerate(labels):
        if l == bg_label:
            continue
        S[idx] = (M==l)
    return S

class Sample(object):

    """
    class for evaluating a singe prediction-gt pair
    """

    def __init__(self, pd, gt, dimension=2, mode='area', tolerance=3, allow_overlap=False, match_method='hungarian'):

        '''
        Args:
            pd: numpy array of dimension D or D+1/list of dimension D
            gt: numpy array of dimension D or D+1/list of dimension D
            dimension: dimension D of the image / ground truth
            mode: 'area' / 'centroid' / 'curve', evaluate area / centroid indicated position / curve
            tolerance: int, shift tolerance, only valid when mode='centroid' / 'curve'
            allow_overlap: if there is no overlap in pd, set to False to save computational cost
            match_method: method used for matching
        Note:
            D + 1 is not supported in 'centroid' mode
            pd/gt can be giveb by:
                - a label map of dimension D, with 0 indicating the background
                - a binary map of demension (D+1) with each instance occupying one channel of the first dimension
            The binary map costs more memory, but can handle overlapped object. If objects are not overlapped, use the label map to save memory and accelarate the computation.
        '''
        self.ndim = dimension
        self.mode = mode
        self.tolerance = tolerance
        self.allow_overlap = allow_overlap
        self.match_method = match_method

        if isinstance(pd, list):
            pd = np.array(pd) if len(pd) != 0 else np.zeros((0,10,10))
        if isinstance(gt, list):
            gt = np.array(gt) if len(gt) != 0 else np.zeros((0,10,10))

        assert (gt.ndim == dimension) or (gt.ndim == dimension+1) or gt.shape[0] == 0
        assert (pd.ndim == dimension) or (pd.ndim == dimension+1) or pd.shape[0] == 0

        if pd.ndim == dimension:
            pd = map2stack(pd)
        if gt.ndim == dimension:
            gt = map2stack(gt)

        # print(pd.shape, gt.shape )

        self.gt, self.pd = gt > 0, pd > 0
        
        # remove 'empty' object in gt, and save size of all objects in gt
        self.S_gt = np.sum(self.gt, axis=tuple(range(1, 1+dimension)))
        self.gt = self.gt[self.S_gt > 0]
        self.S_gt = self.S_gt[self.S_gt>0]
        
        # remove 'empty' object in predcition, and save size of all objects in prediction
        self.S_pd = np.sum(self.pd, axis=tuple(range(1, 1+dimension)))
        self.pd = self.pd[self.S_pd > 0]
        self.S_pd = self.S_pd[self.S_pd>0]

        self.N_gt, self.N_pd = len(self.S_gt), len(self.S_pd)
        self.Intersection = None
        self.Jaccard = None
        self.Dice = None
        self.Match = {}

    def intersection(self):
        '''
        compute the intersection between prediction and ground truth
        Return:
            match: dict of the best match
            intersection: dict of the intersection area
        '''
        
        if self.Intersection is not None:
            return self.Intersection
        
        self.Intersection = np.zeros((self.N_pd, self.N_gt))
        for idx in range(self.N_pd):
            overlap = np.sum(np.multiply(self.gt, np.expand_dims(self.pd[idx], axis=0)), axis=tuple(range(1, 1+self.ndim)))
            self.Intersection[idx] = overlap
        
        self.Dice = self.Intersection * 2 / (np.expand_dims(self.S_pd, axis=1) + np.expand_dims(self.S_gt, axis=0) + 1e-12)
        self.Jaccard = self.Intersection / (np.expand_dims(self.S_pd, axis=1) + np.expand_dims(self.S_gt, axis=0) - self.Intersection + 1e-12)

        return self.Intersection
    
    def match(self, thres):
        '''
        Args:
            thres: threshold to determine the a match
            metric: metric used to determine match, 'Jaccard' or 'Dice'
        Retrun:
            match_count, gt_count: the number of matches, the number of matched gt objects
        '''
        Match = self.Match

        intersection = self.intersection()
        if self.N_gt == 0 or self.N_pd == 0:
            Match[thres] = intersection
        if thres not in Match.keys():
            if (self.allow_overlap or thres < 0.5) and self.match_method is not None:
                if self.match_method == 'hungarian':
                    cost = np.copy(self.Jaccard)
                    cost[cost < thres] = 0
                    idx_pd = np.amax(cost, axis=1) > 0
                    idx_gt = np.amax(cost, axis=0) > 0
                    cost = cost[idx_pd,:][:,idx_gt]

                    match_pd, match_gt = linear_sum_assignment(1-cost)
                    match_pd = np.nonzero(idx_pd)[0][match_pd]
                    match_gt = np.nonzero(idx_gt)[0][match_gt]
                    match = np.zeros(intersection.shape, bool)
                    match[match_pd, match_gt] = True
                if self.match_method == 'mbm':
                    s = MBM_Solver(self.Jaccard >= thres)
                    _, match = s.maxBPM()
                Match[thres] = match
            else:
                Match[thres] = self.Jaccard > thres
        assert np.count_nonzero(Match[thres]) <= self.N_gt and np.count_nonzero(Match[thres]) <= self.N_pd
        return Match[thres]
        
    def averageDice(self, subject='pd'):
        if self.N_gt == 0 and self.N_pd == 0:
            return 1, [1]
        elif self.N_gt == 0 or self.N_pd == 0:
            return 0, [0]
        else:
            max_axis = 1 if subject == 'pd' else 0
            dices = np.amax(self.Dice, axis=max_axis)
            return np.mean(dices), dices

    def averageJaccard(self, subject='pd'):
        if self.N_gt == 0 and self.N_pd == 0:
            return 1, [1]
        elif self.N_gt == 0 or self.N_pd == 0:
            return 0, [0]
        else:
            max_axis = 1 if subject == 'pd' else 0
            jaccards = np.amax(self.Jaccard, axis=max_axis)
            return np.mean(jaccards), jaccards

    def aggregatedJaccard(self):
        '''  
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        intersection = self.intersection()
        if self.N_gt == 0 and self.N_pd == 0:
            return 1, 0, 0
        elif self.N_gt == 0 or self.N_pd == 0:
            return 0, 0, max(np.sum(self.S_gt), np.sum(self.S_pd))
        else:
            idx = np.argmax(intersection, axis=0)
            idx_e = intersection[idx, list(range(self.N_gt))] > 0
            idx_pd, idx_gt = idx[idx_e], np.array(range(self.N_gt))[idx_e]
            
            C = np.sum(intersection[idx_pd, idx_gt])
            U = np.sum(self.S_gt) + np.sum(self.S_pd[idx_pd]) - C + np.sum(self.S_pd[list(set(range(self.N_pd))-set(idx))])

        return C/U, C, U
    
    def AJI(self): # alias of aggregatedJaccard (aggregated Jaccard index)
        return self.aggregatedJaccard()

    def SBD(self):
        '''
        symmetric best dice
        '''
        avgDice1, dices1 = self.averageDice(subject='pd')
        avgDice2, dices2 = self.averageDice(subject='gt')

        if avgDice1 < avgDice2:
            return avgDice1, dices1
        else:
            return avgDice2, dices2

    def detectionRecall(self, thres=0.5):
        match = self.match(thres=thres)
        if self.N_gt != 0:
            N_match = np.sum(match)
            return N_match/self.N_gt, N_match, self.N_gt
        else:
            return 1, 0, 0

    def detectionPrecision(self, thres=0.5):
        match = self.match(thres=thres)
        if self.N_pd != 0:
            N_match = np.sum(match)
            return N_match/self.N_pd, N_match, self.N_pd
        else:
            return 1, 0, 0

    def P_DSB(self, thres=0.5):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        match = self.match(thres=thres)
        N_inter = np.sum(match)
        N_union = self.N_pd + self.N_gt - N_inter
        if N_union != 0:
            return N_inter/N_union, N_inter, N_union 
        else: 
            return 1, 0, 0

    def AP_DSB(self, thres=None):
        '''
        average precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
        ps = [self.P_DSB(thres=t)[0] for t in thres]
        return np.mean(ps)
        
    def RQ(self, thres=0.5):
        '''
        improved recognition quality
        reference: "Panoptic Segmentation" https://arxiv.org/abs/1801.00868
        '''
        match = self.match(thres=thres)
        N_inter = np.sum(match)
        N_union = self.N_pd + self.N_gt
        if N_union != 0:
            return 2*N_inter/N_union, 2*N_inter, N_union 
        else: 
            return 1, 0, 0

    def SQ(self, thres=0.5):
        '''
        improved segmentation quality
        reference: "Panoptic Segmentation" https://arxiv.org/abs/1801.00868
        '''
        if self.N_gt == 0 and self.N_pd == 0:
            return 1, []
        elif self.N_gt == 0 or self.N_pd == 0:
            return 0, []
        else:
            match = self.match(thres=thres)
            rr, cc = np.nonzero(match)
            sqs = self.Jaccard[rr, cc]
            # print(sqs)
            # assert np.all(sqs >= thres)
            sq = np.mean(sqs) if len(sqs) != 0 else 0
            return sq, sqs

    def PQ(self, thres=0.5):
        return self.SQ(thres=thres)[0] * self.RQ(thres=thres)[0]
    
    def mPQ(self, thres=None):
        thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
        pqs = [self.PQ(thres=t) for t in thres]
        return np.mean(pqs)

    def sortedAP(self):
        
        thres = 0.001
        aps = []
        match = self.match(thres=thres)

        TP0 = np.sum(match)
        FN0 = self.N_gt - TP0

        if TP0 == 0 and FN0 != 0:
            return 0, aps, (TP0, FN0)
        elif self.N_pd == 0 and FN0 == 0:
            return 1, aps, (TP0, FN0)
        else:
            jaccard = self.Jaccard[match > 0]
            jaccard_sorted = np.sort(jaccard)
            for k, jac in enumerate(jaccard_sorted):
                ap = (TP0 - k)/(self.N_pd + FN0 + k)
                aps.append((jac, ap))

            score = 0
            jac_pre, ap_pre = 0, aps[0][1]
            for jac, ap in aps:
                score += (jac-jac_pre)*(ap+ap_pre)/2
                jac_pre, ap_pre = jac, ap
            score += (1-jac_pre)*ap_pre/2

            return score, aps, (TP0, FN0)


class MBM_Solver(object):   
    # maximal Bipartite matching. 
    def __init__(self, graph): 
          
        self.graph = graph
        self.persons, self.jobs = graph.shape[0], graph.shape[1]
  
    # A DFS based recursive function that returns true if a matching for vertex u is possible 
    def bpm(self, u, match, seen): 
        for v in range(self.jobs): 
            # If applicant u is interested in job v and v is not seen 
            if self.graph[u][v] and seen[v] == False: 
                seen[v] = True 
                # If job 'v' is not assigned to an applicant OR previously assigned applicant for job v (which is match[v]) has an alternate job available.  
                # Since v is marked as visited in the above line, match[v]  in the following recursive call will not get job 'v' again
                if match[v] == -1 or self.bpm(match[v], match, seen): 
                    match[v] = u 
                    return True
        return False
    
    def maxBPM(self): 
        ''' returns maximum number of matching ''' 
        # applicant number assigned to job i, the value -1 indicates nobody is assigned
        match = [-1] * self.jobs   
        # Count of jobs assigned to applicants 
        N_match = 0 
        for i in range(self.persons): 
            # Mark all jobs as not seen for next applicant. 
            seen = [False] * self.jobs 
            # Find if the applicant 'u' can get a job 
            if self.bpm(i, match, seen): 
                N_match += 1
        match_mx = np.zeros((self.persons, self.jobs), bool)
        for idx_job, idx_person in enumerate(match):
            if idx_person == -1:
                continue
            match_mx[idx_person, idx_job] = True 
        return N_match, match_mx



def evaluator_decorator(metric_name):
    def decorator(fn):
        def decorated(*args,**kwargs):
            kwargs_default = {}
            signature = inspect.signature(fn)
            for k, value in signature.parameters.items():
                if k not in kwargs and value.default != inspect.Signature.empty:
                    kwargs_default[k] = value.default
            if args[0].image_average:
                kwargs_ps = kwargs.copy()
                if 'verbose' in kwargs_ps.keys():
                    del kwargs_ps['verbose']
                metric = []
                for e in args[0].examples:
                    fn_metric = getattr(e, metric_name)
                    # print(args, kwargs_ps)
                    metric.append(fn_metric(*args[1:],**kwargs_ps)[0])
                    # print(metric[-1])
                metric = np.mean(metric)
            else:
                metric = fn(*args,**kwargs)

            verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else kwargs_default['verbose']
            if verbose:
                if metric_name in ['sortedAP']:
                    print('{}: {}, image average: {}'.format(metric_name, metric[0], args[0].image_average))
                else:
                    print('{}: {}, image average: {}'.format(metric_name, metric, args[0].image_average))

            return metric
        return decorated
    return decorator

class Evaluator(object):


    def __init__(self, dimension=2, mode='area', tolerance=3, allow_overlap=False, match_method='hungarian', image_average=False):

        self.ndim = dimension
        self.mode = mode
        self.tolerance = tolerance
        self.allow_overlap = allow_overlap
        self.match_method = match_method
        self.image_average = image_average

        self.examples = []

    def add_example(self, pred, gt, verbose=True):
        e = Sample(pred, gt, dimension=self.ndim, mode=self.mode, tolerance=self.tolerance, allow_overlap=self.allow_overlap, match_method=self.match_method)
        self.examples.append(e)
        if verbose:
            print("example added, total: ", len(self.examples))
    
    def clear(self):
        self.examples = []

    @evaluator_decorator('aggregatedJaccard')
    def aggregatedJaccard(self, verbose=True):
        ''' 
        aggregatedJaccard: accumulate area over images first, then compute the AJI
        meanAggregatedJaccard: compute AJI of each image, and then take the average
        Reference:
            A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology
        '''
        C, U = 0, 0
        for e in self.examples:
            _, C_i, U_i = e.aggregatedJaccard()
            C += C_i
            U += U_i
        if U == 0:
            aj = 1 if C == 0 else 0
        else:
            aj = C/U 
        return aj

    def AJI(self, verbose=True):
        return self.aggregatedJaccard(verbose=verbose)

    @evaluator_decorator('averageDice')
    def averageDice(self, subject='pd', verbose=True):
        dices = []
        for e in self.examples:
            dices.extend(e.averageDice(subject=subject)[1])
        dice = np.mean(dices)
        return dice

    @evaluator_decorator('averageJaccard')
    def averageJaccard(self, subject='pd', verbose=True):
        jaccards = []
        for e in self.examples:
            jaccards.extend(e.averageJaccard(subject=subject)[1])
        jaccard = np.mean(jaccards)
        return jaccard

    @evaluator_decorator('SBD')
    def SBD(self, verbose=True):
        sbd = min(self.averageDice(subject='pd', verbose=False), self.averageDice(subject='gt', verbose=False))
        return sbd 

    @evaluator_decorator('detectionRecall')
    def detectionRecall(self, thres=0.5, verbose=True):
        N_match, N_gt = 0, 0
        for e in self.examples:
            _, N_match_i, N_gt_i = e.detectionRecall(thres=thres)
            N_match += N_match_i
            N_gt += N_gt_i
        # print('recall', N_match, N_gt)
        return N_match/N_gt
            
    @evaluator_decorator('detectionPrecision')
    def detectionPrecision(self, thres=0.5, verbose=True):
        N_match, N_pd = 0, 0
        for e in self.examples:
            _, N_match_i, N_pd_i = e.detectionPrecision(thres=thres)
            N_match += N_match_i
            N_pd += N_pd_i
        # print('precision', N_match, N_pd)
        return N_match/N_pd

    @evaluator_decorator('P_DSB')
    def P_DSB(self, thres=0.5, verbose=True):
        '''
        the precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        N_inter, N_union = 0, 0
        for e in self.examples:
            _, N_inter_i, N_union_i = e.P_DSB(thres=thres)
            N_inter += N_inter_i
            N_union += N_union_i
        # print('ap', N_inter, N_union)
        p = N_inter/N_union if N_union != 0 else 1
        # print(N_inter, N_union)
        return p

    def AP_DSB(self, thres=None, verbose=True):
        '''
        average precision based on Data Scient Bowl 2018 definition: https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
        '''
        thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
        ps = [self.P_DSB(thres=t, verbose=False) for t in thres]
        ap = np.mean(ps)
        if verbose:
            print('AP_DSB: {}, image average: {}'.format(ap, self.image_average))
        return ap

    @evaluator_decorator('RQ')
    def RQ(self, thres=0.5, verbose=True):
        '''
        improved recognition quality
        reference: "Panoptic Segmentation" https://arxiv.org/abs/1801.00868
        '''
        N_inter, N_union = 0, 0
        for e in self.examples:
            _, N_inter_i, N_union_i = e.RQ(thres=thres)
            N_inter += N_inter_i
            N_union += N_union_i    
        rq = N_inter/N_union if N_union != 0 else 1
        return rq

    @evaluator_decorator('SQ')
    def SQ(self, thres=0.5, verbose=True):
        '''
        improved segmentation quality
        reference: "Panoptic Segmentation" https://arxiv.org/abs/1801.00868
        '''
        sq = []
        for e in self.examples:
            sq.extend(e.SQ(thres=thres)[1])
        sq = np.mean(sq) if len(sq) != 0 else 0
        return sq

    def PQ(self, thres=0.5, verbose=True):
        pq = self.SQ(thres=thres, verbose=False) * self.RQ(thres=thres, verbose=False)
        if verbose:
            print('PQ: {}, image average: {}'.format(pq, self.image_average))
        return pq

    def mPQ(self, thres=None, verbose=True):
        thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] if thres is None else thres
        pqs = [self.PQ(thres=t, verbose=False) for t in thres]
        mpq = np.mean(pqs)
        if verbose:
            print('mPQ: {}, image average: {}'.format(mpq, self.image_average))
        return mpq

    @evaluator_decorator('sortedAP')
    def sortedAP(self, truncation=1, verbose=True):

        TP0 = 0
        FN0 = 0
        N_pd = 0

        jaccard = []
        aps = []

        for e in self.examples:
            _, aps_i, (TP0_i, FN0_i) = e.sortedAP()
            jaccard.extend([ja for ja, _ in aps_i])
            TP0 += TP0_i
            FN0 += FN0_i
            N_pd += e.N_pd

        if TP0 == 0 and FN0 != 0:
            return 0
        elif N_pd == 0 and FN0 == 0:
            return 1
        else:
            jaccard_sorted = np.sort(jaccard)
            for k, jac in enumerate(jaccard_sorted):
                ap = (TP0 - k)/(N_pd + FN0 + k)
                if jac > truncation:
                    aps.append((truncation, ap))    
                    break
                else:
                    aps.append((jac, ap))

            score = 0
            jac_pre, ap_pre = 0, aps[0][1]
            for jac, ap in aps:
                score += (jac-jac_pre)*(ap+ap_pre)/2
                jac_pre, ap_pre = jac, ap
            score += (1-jac_pre)*ap_pre/2

            return score, aps



