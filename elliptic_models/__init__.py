"""Reusable methods extracted from the Elliptic exploration notebooks."""
from .data import EllipticData, StratifiedSampler, AnglePreprocessor
from .metrics import Evaluation
from .quantum import QuantumFeatureMap, StatevectorKernel, NystromProjector, KernelClassifier
from .graph import RiskGraph, GraphPeriod, WeisfeilerLehmanKernel
from .experiments import ClassificationExperiment, GraphExperiment, RBFKernel
from .hardware import IBMFidelityKernel

__all__ = ['EllipticData', 'StratifiedSampler', 'AnglePreprocessor', 'Evaluation',
           'QuantumFeatureMap', 'StatevectorKernel', 'NystromProjector', 'KernelClassifier',
           'RiskGraph', 'GraphPeriod', 'WeisfeilerLehmanKernel', 'ClassificationExperiment',
           'GraphExperiment', 'RBFKernel', 'IBMFidelityKernel']

from .multihop import DirectionalPropagation, HopFeatureMap
from .topology import (TopologySample, RootedSubgraphSampler, TopologyPreprocessor,
                       TopologyFeatureMap, TopologyDiagnostics, PermutationAveragedKernel)
from .graph_studies import KernelComparison, PropagationStudy, TopologyStudy

__all__ += ['DirectionalPropagation', 'HopFeatureMap', 'TopologySample',
            'RootedSubgraphSampler', 'TopologyPreprocessor', 'TopologyFeatureMap',
            'TopologyDiagnostics', 'PermutationAveragedKernel', 'KernelComparison',
            'PropagationStudy', 'TopologyStudy']

from .reporting import ResultReport, ResultPlots
__all__ += ['ResultReport', 'ResultPlots']
from .reporting import ExperimentSummary
__all__ += ['ExperimentSummary']

from .additive import FullGraphInputs, AdditiveKernelClassifier
__all__ += ['FullGraphInputs', 'AdditiveKernelClassifier']

from .comparisons import MatchedComparisons
__all__ += ['MatchedComparisons']

from .topology_encoders import GraphMessageFeatureMap, GraphHamiltonianFeatureMap, HermitianWalkFeatureMap
from .topology_research import GraphDescriptors, StructuralProbeStudy, GraphRetention, TopologyEncodingStudy
from .story import TopologyResearchPlots, StorySummary
__all__ += ['GraphMessageFeatureMap', 'GraphHamiltonianFeatureMap', 'HermitianWalkFeatureMap',
            'GraphDescriptors', 'StructuralProbeStudy', 'GraphRetention', 'TopologyEncodingStudy',
            'TopologyResearchPlots', 'StorySummary']

from .topology_encoders import HamiltonianMomentKernel
__all__ += ['HamiltonianMomentKernel']

from .hardware_experiments import HamiltonianHardwareExperiment
__all__ += ['HamiltonianHardwareExperiment']

from .experiment_flow import ExperimentFlow, PairedSelection, ResultStore
__all__ += ['ExperimentFlow', 'PairedSelection', 'ResultStore']

from .four_cases import FourCaseStudy
__all__ += ['FourCaseStudy']
