"""Genetic Algorithm optimiser module."""
from typing import Tuple, Callable, Optional
import sys
import time
import numpy as np
try:
    from geneticalgorithm import geneticalgorithm
except ImportError:
    # Show a nice exception when this package is used
    from piglot.optimiser import missing_method
    geneticalgorithm = missing_method("Genetic algorithm", "geneticalgorithm")
from piglot.objective import Objective
from piglot.optimiser import ScalarOptimiser


class geneticalgorithmMod(geneticalgorithm):

    def run(self, optimiser, init_shot):
        # Initial Population
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)
        for p in range(0,self.pop_s):
            for i in self.integers[0]:
                var[i]=np.random.randint(self.var_bound[i][0],
                        self.var_bound[i][1]+1)
                solo[i]=var[i].copy()
            for i in self.reals[0]:
                var[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])
                solo[i]=var[i].copy()
            # Use initial shot
            if p==0:
                var = init_shot
                solo[0:self.dim] = var.copy()
            obj=self.sim(var)
            solo[self.dim]=obj
            pop[p]=solo.copy()

        # Report
        self.report=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        t=1
        counter=0
        while t<=self.iterate:
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            # Report

            self.report.append(pop[0,self.dim])

            # Normalizing objective function
            normobj=np.zeros(self.pop_s)
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
            else:
                normobj=pop[:,self.dim].copy()
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            # Calculate probability
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)

            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
            ef_par=par[ef_par_list].copy()

            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)
                solo[: self.dim]=ch1.copy()
                obj=self.sim(ch1)
                solo[self.dim]=obj
                pop[k]=solo.copy()
                solo[: self.dim]=ch2.copy()
                obj=self.sim(ch2)
                solo[self.dim]=obj
                pop[k+1]=solo.copy()
            # Check progress
            if optimiser._progress_check(t, self.best_function, self.best_variable):
                break
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True

        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        if pop[0,self.dim]<self.best_function:
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        # Report
        self.report.append(pop[0,self.dim])
        self.output_dict={'variable': self.best_variable, 'function': self.best_function}
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
        #sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        #sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        #sys.stdout.flush()
        re=np.array(self.report)
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+
                             ' maximum number of iterations without improvement was met!')


class GA(ScalarOptimiser):
    """
    Genetic Algorithm optimiser.
    Documentation: https://pypi.org/project/geneticalgorithm/

    Attributes
    ----------
    variable_type : string
        'bool' if all variables are Boolean;
        'int' if all variables are integer;
        and 'real' if all variables are real value or continuous (for mixed type see
        parameter variable_type_mixed)
    variable_type_mixed : numpy array/None
        Default None; leave it None if all variables have the same type;
        otherwise this can be used to specify the type of each variable separately.
        For example if the first variable is integer but the second one is real the
        input is: np.array(['int'],['real']). NOTE: it does not accept 'bool'.
        If variable type is Boolean use 'int' and provide a boundary as [0,1] in
        variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.
    function_timeout : float
        if the given function does not provide output before function_timeout (unit is
        seconds) the algorithm raise error. For example, when there is an infinite
        loop in the given function.
    algorithm_parameters : dictionary
        Algorithm parameters.
    convergence_curve : True/False
        Plot the convergence curve or not. Default is True.
    progress_bar : True/False
        Show progress bar or not. Default is True.


    Methods
    -------
    _optimise(self, func, n_dim, n_iter, bound, init_shot):
        Solves the optimization problem
    """

    def __init__(
            self,
            objective: Objective,
            variable_type='real',
            variable_type_mixed=None,
            function_timeout=3600,
            algorithm_parameters={
                'max_num_iteration': None,
                'population_size': 100,
                'mutation_probability': 0.1,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type': 'uniform',
                'max_iteration_without_improv': None,
            },
            convergence_curve=True,
            progress_bar=False,
            ):
        """
        Constructs all the necessary attributes for the Genetic Algorithm optimiser

        Parameters
        ----------
        objective : Objective
            Objective function to optimise.
        variable_type : string
            'bool' if all variables are Boolean;
            'int' if all variables are integer;
            and 'real' if all variables are real value or continuous (for mixed type see
            parameter variable_type_mixed)
        variable_type_mixed : numpy array/None
            Default None; leave it None if all variables have the same type;
            otherwise this can be used to specify the type of each variable separately.
            For example if the first variable is integer but the second one is real the
            input is: np.array(['int'],['real']). NOTE: it does not accept 'bool'.
            If variable type is Boolean use 'int' and provide a boundary as [0,1] in
            variable_boundaries. Also if variable_type_mixed is applied,
            variable_boundaries has to be defined.
        function_timeout : float
            if the given function does not provide output before function_timeout (unit is
            seconds) the algorithm raise error. For example, when there is an infinite
            loop in the given function.
        algorithm_parameters : dictionary
            max_num_iteration : int
            population_size : int
            mutation_probability : float in [0,1]
            elit_ration : float in [0,1]
            crossover_probability : float in [0,1]
            parents_portion : float in [0,1]
            crossover_type : string
                Default is 'uniform'; 'one_point' or two_point' are other options
            max_iteration_without_improv : int
                maximum number of successive iterations without improvement. If None it is
                ineffective
        convergence_curve : True/False
            Plot the convergence curve or not. Default is True.
        progress_bar : True/False
            Show progress bar or not. Default is True.
        """
        super().__init__('GA', objective)
        self.variable_type = variable_type
        self.variable_type_mixed = variable_type_mixed
        self.function_timeout = function_timeout
        self.algorithm_parameters = algorithm_parameters
        self.convergence_curve = convergence_curve
        self.progress_bar = progress_bar

    def _scalar_optimise(
        self,
        objective: Callable[[np.ndarray, Optional[bool]], float],
        n_dim: int,
        n_iter: int,
        bound: np.ndarray,
        init_shot: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Abstract method for optimising the objective.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to optimise.
        n_dim : int
            Number of parameters to optimise.
        n_iter : int
            Maximum number of iterations.
        bound : np.ndarray
            Array where first and second columns correspond to lower and upper bounds, respectively.
        init_shot : np.ndarray
            Initial shot for the optimisation problem.

        Returns
        -------
        float
            Best observed objective value.
        np.ndarray
            Observed optimum of the objective.
        """
        self.algorithm_parameters['max_num_iteration'] = n_iter
        model = geneticalgorithmMod(objective, n_dim, self.variable_type, bound,
                                    self.variable_type_mixed,
                                    self.function_timeout, self.algorithm_parameters,
                                    self.convergence_curve, self.progress_bar)
        model.run(self, init_shot)
        return model.output_dict.get('variable'), model.output_dict.get('function')
