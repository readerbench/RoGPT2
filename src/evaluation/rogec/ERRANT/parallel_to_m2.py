import argparse
import os
import spacy
import scripts.align_text as align_text
import scripts.toolbox as toolbox
def main(args):
	# Get base working directory.
	basename = os.path.dirname(os.path.realpath(__file__))
	print("Loading resources...")
	spacy_disable = ['ner']
	treetagger = None
	lang = None
	if args.lang == "en":
		from nltk.stem.lancaster import LancasterStemmer
		import scripts.cat_rules as cat_rules
		# Load Tokenizer and other resources
		nlp = spacy.load("en", disable=spacy_disable)
		# Lancaster Stemmer
		stemmer = LancasterStemmer()
		# GB English word list (inc -ise and -ize)
		word_list = toolbox.loadDictionary(basename+"/resources/en_GB-large.txt")
		# Part of speech map file
		tag_map = toolbox.loadTagMap(basename+"/resources/en-ptb_map", args)
	elif args.lang == "de":
		from nltk.stem.snowball import GermanStemmer
		import scripts.cat_rules_de as cat_rules
		import treetaggerwrapper
		treetagger = treetaggerwrapper.TreeTagger(TAGLANG="de",TAGDIR=basename+"/resources/tree-tagger-3.2")
		# Load Tokenizer and other resources
		nlp = spacy.load("de", disable=spacy_disable)
		# German Snowball Stemmer
		stemmer = GermanStemmer()
		# DE word list from hunspell
		word_list = toolbox.loadDictionary(basename+"/resources/de_DE-large.txt")
		# Part of speech map file (not needed for spacy 2.0)
		tag_map = toolbox.loadTagMap(basename+"/resources/de-stts_map", args)
	elif args.lang == "ro":
		from nltk.stem.snowball import RomanianStemmer
		from rb.parser.spacy_parser import SpacyParser
		import scripts.cat_rules_ro as cat_rules
		# no tagger, we use only pos & lemmas from the spacy model
		treetagger, tag_map = None, None
		lang = 'ro'
		# Load Tokenizer and other resources
		nlp = SpacyParser.get_instance()
		# Romanian Snowball Stemmer
		stemmer = RomanianStemmer()
		# RO word list from hunspell
		word_list = toolbox.loadDictionary(basename+"/resources/ro_RO-large.txt")

	# Setup output m2 file
	out_m2 = open(args.out, "w")

	print("Processing files...")
	nr_edits, nr_edits_per_sentence = 0, 0
	length_edits_per_sentence, length_edits = 0, 0
	nr_edits_per_category = {'POS': 0}
	total_tokens = 0
	
	# Open the original and corrected text files.
	with open(args.orig) as orig, open(args.cor) as cor:
		# Process each pre-aligned sentence pair.
		count_sentences = 0
		for orig_sent, cor_sent in zip(orig, cor):
			count_sentences += 1
			total_tokens += len(orig_sent.split(' '))
			# Markup the parallel sentences with spacy
			proc_orig = toolbox.applySpacy(orig_sent, nlp, args, treetagger, lang=lang)
			proc_cor = toolbox.applySpacy(cor_sent, nlp, args, treetagger, lang=lang)
			# Write the original sentence to the output m2 file.
			proc_orig_tokens = [token.text for token in proc_orig]
			out_m2.write("S "+ " ".join(proc_orig_tokens))
			# Identical sentences have no edits, so just write noop.
			if orig_sent.strip() == cor_sent.strip():
				out_m2.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
			# Otherwise, do extra processing.
			else:
				# Output the annotations for orig/cor
				if (args.ann):
				  out_m2.write("O "+toolbox.formatAnnotation(proc_orig)+"\n")
				  out_m2.write("C "+toolbox.formatAnnotation(proc_cor)+"\n")
				# Auto align the parallel sentences and extract the edits.
				auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, nlp, args, language=lang)
				# Loop through the edits.
				nr_edits_per_sentence += len(auto_edits)
				nr_edits += len(auto_edits)
				for auto_edit in auto_edits:
					length_edit = max(auto_edit[1] - auto_edit[0], len(auto_edit[3].split(' ')))
					length_edits += length_edit
					length_edits_per_sentence += length_edit
					# Give each edit an automatic error type.
					cat = cat_rules.autoTypeEdit(auto_edit, proc_orig, proc_cor, word_list, tag_map, nlp, stemmer)[2:]
					if not (cat == 'MORPH' or cat == 'ORTH' or cat == 'SPELL'
						 or cat == 'ORDER' or cat == 'CONTR' or cat == 'OTHER'):
						 nr_edits_per_category['POS'] += 1
					nr_edits_per_category[cat] = 1 + nr_edits_per_category[cat] if cat in nr_edits_per_category else 1
					auto_edit[2] = cat
					# Write the edit to the output m2 file.
					out_m2.write(toolbox.formatEdit(auto_edit)+"\n")
			# Write a newline when there are no more edits.
			out_m2.write("\n")

		nr_edits_per_sentence /= count_sentences
		length_edits_per_sentence /= count_sentences
		length_edits /= nr_edits
		if args.stats:
			print(f'nr_edits_per_sent: {nr_edits_per_sentence}')
			print(f'length_edits_per_sentence: {length_edits_per_sentence}')
			print(f'nr_edits_per_category: {nr_edits_per_category}')
			print(f'length_edits: {length_edits}')

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser(description="Convert parallel original and corrected text files (1 sentence per line) into M2 format.\nThe default uses Damerau-Levenshtein and merging rules and assumes tokenized text.",
								formatter_class=argparse.RawTextHelpFormatter,
								usage="%(prog)s [-h] [options] -orig ORIG -cor COR -out OUT")
	parser.add_argument("-orig", help="The path to the original text file.", required=True)
	parser.add_argument("-cor", help="The path to the corrected text file.", required=True)
	parser.add_argument("-out",	help="The output filepath.", required=True)						
	parser.add_argument("-lang", choices=["en", "de", "ro"], default="en", help="Input language. Currently supported: en (default), de, ro\n")
	parser.add_argument("-lev",	help="Use standard Levenshtein to align sentences.", action="store_true")
	parser.add_argument("-merge", choices=["rules", "all-split", "all-merge", "all-equal"], default="rules",
						help="Choose a merging strategy for automatic alignment.\n"
								"rules: Use a rule-based merging strategy (default)\n"
								"all-split: Merge nothing; e.g. MSSDI -> M, S, S, D, I\n"
								"all-merge: Merge adjacent non-matches; e.g. MSSDI -> M, SSDI\n"
								"all-equal: Merge adjacent same-type non-matches; e.g. MSSDI -> M, SS, D, I")
	parser.add_argument("-tok", help="Tokenize input using spacy tokenizer.", action="store_true")
	parser.add_argument("-ann", help="Output automatic annotation.", action="store_true")
	parser.add_argument("-stats", help="Print stats", action="store_true")
	args = parser.parse_args()
	main(args)
	