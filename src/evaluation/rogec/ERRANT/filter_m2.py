import argparse
import os
import spacy
import scripts.align_text as align_text
import scripts.toolbox as toolbox
import regex as re

def main(args):
	# Setup output m2 file
	out_m2 = open(args.out, "w")

	print("Processing files...")
	# Open the m2 file and split into sentence+edit chunks.
	m2_file = open(args.ref).read().strip().split("\n\n")
	ref_edit_profile_list = set()
	ref_edit_profile_set = set()
	ref_token_profile = set()
	for info in m2_file:
		# Get the original and corrected sentence + edits for each annotator.
		orig_sent, coder_dict = toolbox.processM2(info)
		# Save info about types of edit groups seen
		# Only process sentences with edits.
		if coder_dict:
			# Save marked up original sentence here, if required.
			proc_orig = ""
			# Loop through the annotators
			edit_list = []
			edit_set = set()
			token_set = set()
			for coder, coder_info in sorted(coder_dict.items()):
				cor_sent = coder_info[0]
				gold_edits = coder_info[1]
				# If there is only 1 edit and it is noop, do nothing
				if gold_edits[0][2] == "noop":
					#out_m2.write(toolbox.formatEdit(gold_edits[0], coder)+"\n")				
					continue
				# Loop through gold edits.
				for gold_edit in gold_edits:
					if orig_sent[gold_edit[0]:gold_edit[1]]:
						token_set.add(" ".join(orig_sent[gold_edit[0]:gold_edit[1]]))
					edit_set.add(gold_edit[2])
					edit_list.append(gold_edit[2])
			if len(edit_set) > 0:
				edit_list.sort()
				ref_edit_profile_list.add(tuple(edit_list))
				ref_edit_profile_set.add(frozenset(edit_set))
			ref_token_profile = ref_token_profile | token_set

	skip_edit_sets = set(['R:OTHER', 'M:OTHER', 'U:OTHER', 
			'R:PUNCT', 'M:PUNCT', 'U:PUNCT',
			'R:PNOUN', 'M:PNOUN', 'U:PNOUN'])

	m2_filter_file = open(args.filter).read().strip().split("\n\n")
	for info in m2_filter_file:
		# Get the original and corrected sentence + edits for each annotator.
		orig_sent, coder_dict = toolbox.processM2(info)
    # decide whether to keep this sentence
		# Only process sentences with edits.
		if coder_dict:
			# Loop through the annotators
			edit_list = []
			edit_set = set()
			token_set = set()
			long_other_edits = False
			no_latin_chars = False
			keep = False
			for coder, coder_info in sorted(coder_dict.items()):
				cor_sent = coder_info[0]
				gold_edits = coder_info[1]
				# If there is only 1 edit and it is noop, do nothing
				if gold_edits[0][2] == "noop":
					continue
				# Loop through gold edits.
				for gold_edit in gold_edits:
					if orig_sent[gold_edit[0]:gold_edit[1]]:
						token_set.add(" ".join(orig_sent[gold_edit[0]:gold_edit[1]]))
					edit_set.add(gold_edit[2])
					edit_list.append(gold_edit[2])

					if ":OTHER" in gold_edit[2]:
						gold_len = gold_edit[1] - gold_edit[0]
						cor_len = len(gold_edit[3].split(" "))
						if abs(gold_len - cor_len) > 2:
							long_other_edits = True
						elif gold_len > 3 or cor_len > 3:
							long_other_edits = True

			edit_list.sort()
			
			if args.debug:
				print()
				print("INPUT ORIG:       ", orig_sent)
				print("INPUT COR:        ", cor_sent)
				print("INPUT EDITS:      ", gold_edits)
				print("INPUT EDIT TYPES: ", edit_list)
				print("INPUT EDIT TOKENS:", token_set)
			# skip if edits subset of skip list
			if edit_set <= skip_edit_sets:
				if args.debug:
					print("SKIPLIST", edit_list)
				pass
			# keep if no skip edits
			elif len(edit_set & skip_edit_sets) == 0:
				keep = True
			# skip sentences with no Latin characters in corrections
			elif not re.search('\p{Latin}', ''.join(token_set)):
				if args.debug:
					print("NOLATIN", edit_list)
				pass
			# skip sentences with formulas
			elif "formula_" in ''.join(orig_sent) or "formula_" in ''.join(cor_sent):
				if args.debug:
					print("FORMULA", edit_list)
				pass
			# discard if any OTHER longer than two tokens:
			elif long_other_edits > 0:	
				if args.debug:
					print("LONG", edit_list)
				pass
			# only differences are numbers
			elif re.match("^\d+$", ''.join(token_set)):
				if args.debug:
					print("NUM", edit_list)
				pass
			# keep remaining edit lists seen in ref
			elif tuple(edit_list) in ref_edit_profile_list:
				keep = True
			# keep remaining single edits
			elif len(edit_list) == 1:
				keep = True
			# keep remaining set seen in ref with 2+ edits
			elif len(edit_list) > 1 and edit_set in ref_edit_profile_set:
				keep = True
			# keep edit if token seen in ref
			elif len(token_set & ref_token_profile) > 0 and len(token_set & ref_token_profile) > len(token_set) - 1:
				keep = True
			else:
				# keep if set similarity > threshold
				for ref_edit_set in ref_edit_profile_set:
					if len(ref_edit_set & edit_set) / float(len(ref_edit_set | edit_set)) > 0.5:
						keep = True
						break

				if not keep and args.debug:
					print("OTHER", edit_list)

			if keep:
				out_m2.write("S "+" ".join(orig_sent)+"\n")
				for gold_edit in gold_edits:
					out_m2.write(toolbox.formatEdit(gold_edit, coder)+"\n")
				out_m2.write("\n")

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser(description="Filter an M2 file based on edits in a reference M2 file (both generated with identical settings with ERRANT).",
								formatter_class=argparse.RawTextHelpFormatter,
								usage="%(prog)s [-h] [options] -ref REF -filter FILTER -out OUT")
	parser.add_argument("-ref", help="The reference M2 file.", required=True)
	parser.add_argument("-filter", help="The M2 file to filter.", required=True)
	parser.add_argument("-out",	help="The output filepath to the filtered M2 file.", required=True)		
	parser.add_argument("-lang", choices=["en", "de"], default="en", help="Input language. Currently supported: en (default), de\n")
	parser.add_argument("-debug", dest='debug', action='store_true', help="Debugging output.\n")
	args = parser.parse_args()
	args.ann = False
	args.tok = False
	main(args)
