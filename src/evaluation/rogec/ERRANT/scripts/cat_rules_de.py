from difflib import SequenceMatcher
from string import punctuation
import spacy.parts_of_speech as spos

# Contractions
conts = {"'s", "´s"} 
# Rare POS tags that make uninformative error categories
rare_tags = {"INTJ", "NUM", "SYM", "X"}
# POS tags with inflectional morphology
inflected_tags = {"ADJ", "ADV", "AUX", "DET", "PRON", "PROPN", "NOUN", "VERB"}
# Fine-grained STTS verb tags
verb_tags = {"VVINF", "VVIZU", "VVPP", "VAINF", "VAPP", "VMINF", "VMPP", "VVFIN", "VVIMP", "VAFIN", "VAIMP", "VMFIN", "VMPP"}
# Some dep labels that map to pos tags.
dep_map = {"ac": "ADP",
	"svp": "ADP",
	"punct": "PUNCT" }

# Input 1: An edit list. [orig_start, orig_end, cat, cor, cor_start, cor_end]
# Input 2: An original SpaCy sentence.
# Input 3: A corrected SpaCy sentence.
# Input 4: A set of valid words.
# Input 5: A dictionary to map detailed tags to Stanford Universal Dependency tags.
# Input 6: A preloaded spacy processing object.
# Input 7: The stemmer.
# Output: The input edit with new error tag, in M2 edit format.
def autoTypeEdit(edit, orig_sent, cor_sent, word_list, tag_map, nlp, stemmer):
	# Get the tokens in the edit.
	orig_toks = orig_sent[edit[0]:edit[1]]
	cor_toks = cor_sent[edit[4]:edit[5]]
	# Nothing to nothing is a detected, but not corrected edit.
	if not orig_toks and not cor_toks:
		return "UNK"
	# Missing
	elif not orig_toks and cor_toks:
		op = "M:"
		cat = getOneSidedType(cor_toks, tag_map)
	# Unnecessary
	elif orig_toks and not cor_toks:
		op = "U:"
		cat = getOneSidedType(orig_toks, tag_map)
	# Replacement and special cases
	else:
		# Same to same is a detected, but not corrected edit.
		if orig_toks.text == cor_toks.text:
			return "UNK"
		# Special: Orthographic errors at the end of multi-token edits are ignored.
		# E.g. [Doctor -> The doctor], [The doctor -> Dcotor], [, since -> . Since]
		# Classify the edit as if the last token weren't there.
		elif orig_toks[-1].lower_ == cor_toks[-1].lower_ and \
			(len(orig_toks) > 1 or len(cor_toks) > 1):
			min_edit = edit[:]
			min_edit[1] -= 1
			min_edit[5] -= 1
			return autoTypeEdit(min_edit, orig_sent, cor_sent, word_list, tag_map, nlp, stemmer)
		# Replacement
		else:
			op = "R:"
			cat = getTwoSidedType(orig_toks, cor_toks, word_list, tag_map, nlp, stemmer)
	return op+cat

# Input 1: Spacy tokens
# Input 2: A map dict from detailed to universal dependency pos tags.
# Output: A list of token, pos and dep tag strings.
def getEditInfo(toks, tag_map):
	text = []
	pos = []
	dep = []
	for tok in toks:
		text.append(tok.text)
		pos.append(tag_map[tok.tag_])
		dep.append(tok.dep_)
	return text, pos, dep

# Input 1: Spacy tokens.
# Input 2: A map dict from detailed to universal dependency pos tags.
# Output: An error type string.
# When one side of the edit is null, we can only use the other side.
def getOneSidedType(toks, tag_map):
	# Extract strings, pos tags and parse info from the toks.
	str_list, pos_list, dep_list = getEditInfo(toks, tag_map)

	# Special cases.
	if len(toks) == 1:
		# Contraction.
		if toks[0].lower_ in conts:
			return "CONTR"
	# POS-based tags. Ignores rare, uninformative categories.
	if len(set(pos_list)) == 1 and pos_list[0] not in rare_tags:
		return pos_list[0]
	# More POS-based tags using special dependency labels.
	if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
		return dep_map[dep_list[0]]
	# zu-infinitives
	if set(pos_list) == {"PART", "VERB"}:
		return "VERB"
	# Tricky cases
	else:
		return "OTHER"

# Input 1: Original text spacy tokens.
# Input 2: Corrected text spacy tokens.
# Input 3: A set of valid words.
# Input 4: A map from detailed to universal dependency pos tags.
# Input 5: A preloaded spacy processing object.
# Input 6: The stemmer in NLTK.
# Output: An error type string.
def getTwoSidedType(orig_toks, cor_toks, word_list, tag_map, nlp, stemmer):
	# Extract strings, pos tags and parse info from the toks.
	orig_str, orig_pos, orig_dep = getEditInfo(orig_toks, tag_map)
	cor_str, cor_pos, cor_dep = getEditInfo(cor_toks, tag_map)

	# Orthography; i.e. whitespace and/or case errors.
	if onlyOrthChange(orig_str, cor_str):
		return "ORTH"
	# Word Order; only matches exact reordering.
	if exactReordering(orig_str, cor_str):
		return "WO"

	# 1:1 replacements (very common)
	if len(orig_str) == len(cor_str) == 1:
		# 2. SPELLING AND INFLECTION
		# Only check alphabetical strings on the original side.
		# Spelling errors take precedence over POS errors so this rule is ordered.
		if orig_str[0].isalpha():
			# Check a dict for both orig and lower case.
			# "cat" is in the dict, but "Cat" is not.
			if orig_str[0] not in word_list and orig_str[0].lower() not in word_list:
				# Check if both sides have a common lemma
				if sameLemma(orig_toks[0], cor_toks[0], nlp):
					# skip to morphology
					pass
				# Use string similarity to detect true spelling errors.
				else:
					char_ratio = SequenceMatcher(None, orig_str[0], cor_str[0]).ratio()
					# Ratio > 0.5 means both side share at least half the same chars.
					# WARNING: THIS IS AN APPROXIMATION.
					if char_ratio > 0.5:
						return "SPELL"
					# If ratio is <= 0.5, this may be a spelling+other error; e.g. tolk -> say
					else:
						# If POS is the same, this takes precedence over spelling.
						if orig_pos == cor_pos and orig_pos[0] not in rare_tags:
							return orig_pos[0]
						# Tricky cases.
						else:
							return "OTHER"

		# 3. MORPHOLOGY
		# Only DET, ADJ, ADV, NOUN and VERB with same lemma can have inflectional changes.
		if sameLemma(orig_toks[0], cor_toks[0], nlp) and \
			orig_pos[0] in inflected_tags and cor_pos[0] in inflected_tags:
			# Same POS on both sides
			if orig_pos == cor_pos:
				# Inflection
				if orig_pos[0] in inflected_tags:
					return orig_pos[0] + ":FORM"
			# For remaining verb errors, rely on cor_pos
			if cor_toks[0].tag_ in verb_tags:
				return "VERB:FORM"
			# Tricky cases that all have the same lemma.
			else:
				return "MORPH"
		# Inflectional morphology.
		if stemmer.stem(orig_str[0]) == stemmer.stem(cor_str[0]) and \
			orig_pos[0] in inflected_tags and cor_pos[0] in inflected_tags:
			return "MORPH"

		# 4. GENERAL
		# POS-based tags. Some of these are context sensitive mispellings.
		if orig_pos == cor_pos and orig_pos[0] not in rare_tags:
			return orig_pos[0]
		# Some dep labels map to POS-based tags.
		if orig_dep == cor_dep and orig_dep[0] in dep_map.keys():
			return dep_map[orig_dep[0]]
		# Separable verb prefixes vs. prepositions
		if set(orig_pos+cor_pos) == {"ADP"} or set(orig_dep+cor_dep) == {"ac", "svp"}:
			return "ADP"
		# Can use dep labels to resolve DET + PRON combinations.
		if set(orig_pos+cor_pos) == {"DET", "PRON"}:
			# DET cannot be a subject or object.
			if cor_dep[0] in {"sb", "oa", "od", "og"}:
				return "PRON"
			# "poss" indicates possessive determiner
			if cor_dep[0] == "ag":
				return "DET"
		# Tricky cases.
		else:
			return "OTHER"

	# All same POS
	if len(set(orig_pos+cor_pos)) == 1 and orig_pos[0] not in rare_tags:
		return orig_pos[0]
	# All same special dep labels.
	if len(set(orig_dep+cor_dep)) == 1 and orig_dep[0] in dep_map.keys():
		return dep_map[orig_dep[0]]
	# Infinitives, gerunds, phrasal verbs.
	if set(orig_pos+cor_pos) == {"PART", "VERB"}:
		# Final verbs with the same lemma are form; e.g. zu essen -> essen
		if sameLemma(orig_toks[-1], cor_toks[-1], nlp):
			return "VERB:FORM"
		# Remaining edits are often verb; e.g. to eat -> consuming, look at -> see
		else:
			return "VERB"

	# Tricky cases.
	else:
		return "OTHER"

# Input 1: A list of original token strings
# Input 2: A list of corrected token strings
# Output: Boolean; the difference between the inputs is only whitespace or case.
def onlyOrthChange(orig_str, cor_str):
	orig_join = "".join(orig_str).lower()
	cor_join = "".join(cor_str).lower()
	if orig_join == cor_join:
		return True
	return False

# Input 1: A list of original token strings
# Input 2: A list of corrected token strings
# Output: Boolean; the tokens are exactly the same but in a different order.
def exactReordering(orig_str, cor_str):
	# Sorting lets us keep duplicates.
	orig_set = sorted([tok.lower() for tok in orig_str])
	cor_set = sorted([tok.lower() for tok in cor_str])
	if orig_set == cor_set:
		return True
	return False

# Input 1: An original text spacy token.
# Input 2: A corrected text spacy token.
# Input 3: A spaCy processing object.
# Output: Boolean; the tokens have the same lemma.
# Spacy only finds lemma for its predicted POS tag. Sometimes these are wrong,
# so we also consider alternative POS tags to improve chance of a match.
def sameLemma(orig_tok, cor_tok, nlp):
	if orig_tok.lemma_ == cor_tok.lemma_:
		return True
	return False
