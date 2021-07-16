import argparse
import os
import spacy
import scripts.align_text as align_text
import scripts.toolbox as toolbox
import regex as re

def main(args):
	# Setup output m2 file
	out_parallel = open(args.out, "w")

	print("Processing files...")
	# Open the m2 file and split into sentence+edit chunks.
	m2_file = open(args.m2).read().strip().split("\n\n")
	for info in m2_file:
		# Get the original and corrected sentence + edits for each annotator.
		orig_sent, coder_dict = toolbox.processM2(info)
		# Save info about types of edit groups seen
		# Only process sentences with edits.
		if coder_dict:
			# Save marked up original sentence here, if required.
			proc_orig = ""
			# Loop through the annotators
			for coder, coder_info in sorted(coder_dict.items()):
				cor_sent = coder_info[0]
				out_parallel.write(" ".join(orig_sent) + "\t" + " ".join(cor_sent) + "\n")

	out_parallel.close()

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser(description="Filter an M2 file based on edits in a reference M2 file (both generated with identical settings with ERRANT).",
								formatter_class=argparse.RawTextHelpFormatter,
								usage="%(prog)s [-h] [options] -m2 M2 -out OUT")
	parser.add_argument("-m2", help="The M2 file.", required=True)
	parser.add_argument("-out",	help="The output filepath to the parallel file.", required=True)		
	args = parser.parse_args()
	main(args)
