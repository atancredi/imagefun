class PipelineBuilder:

	def run_function(self, func, **kwargs):
		func(self, **kwargs)
		return self

	# CONDITIONAL
	def run_if_condition(self, condition, function):
		if condition:
			function(self)
		return self
	