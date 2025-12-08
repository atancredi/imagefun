from typing import Callable, Self, List, Any
from logging import Logger

class PipelineBuilder:

	def set_logger(self, logger: Logger):
		self.logger = logger
		return self
	
	def run_function(self, func: Callable[[Self], Self], **kwargs):
		func(self, **kwargs)
		return self

	# CONDITIONAL
	def run_if_condition(self, condition, function: Callable[[Self], Self]):
		if condition:
			function(self)
		return self

	# ITERATIVE
	def run_iterations(self, parameter_list: List[Any], function: Callable[[Self], Self]):
		for parameter in parameter_list:
			function(self, parameter)
		return self
	
	