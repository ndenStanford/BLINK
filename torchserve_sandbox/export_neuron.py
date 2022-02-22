import torch
import torch_neuron
import numpy as np

model = torch.jit.load("blink_entity_linking.pt")
compiler_options = "-O2"
optimize = "aggressive"

cand_encs = torch.load('sample_input/cand_encs_cpu.pt')
text_vecs = torch.load('sample_input/text_vecs_cpu.pt')
inputs = [text_vecs, cand_encs]

module_neuron = torch_neuron.trace(model, example_inputs=inputs, compiler_args=compiler_options, optimize = optimize)
module_neuron.save("blink_entity_linking_neuron.pt")

#module_loaded_neuron=torch.jit.load("blink_enlity_linking_neuron.pt")
#print(module_loaded_neuron)