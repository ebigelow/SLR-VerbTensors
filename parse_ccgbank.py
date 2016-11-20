# CCGBank formatting, see pg. 134:
#   http://nlp.cs.illinois.edu/HockenmaierGroup/Papers/CCGbank/CCGbankManual.pdf
#
#   format is modified with {  } tags at the end, which 
#   indicate forward/backward function application
#
#   {FA, OTHER}     TODO  what is 'other'?
#

def parse_node(s):
    """
    Input: the stuff between < and >

    """
    sp = s.split(' ')
    if sp[0] == 'L':
        word, tag = sp[-2:]
        return ParseNode(word, tag, 0)

    else:
        _, tag, head, args = sp
        return ParseNode(None, tag, int(args), int(head))


def parse_tree(s):
    if s[0] == '(' and s[-1] == ')': s = s[1:-1]

    # Generate node with initial info
    i1 = s.find('<')+1
    i2 = s.find('>')
    node = parse_node(s[i1:i2])

    # Generate nodes for children
    if node.args >= 1:
        s_ = s[i2+1:]
        ia1, ia2 = find_parens(s_, '(', ')')
        node.child1 = parse_tree(s_[ia1+1:ia2])
    if node.args == 2:
        s_ = s_[ia2+1:]
        ia3, ia4 = find_parens(s_, '(', ')')
        node.child2 = parse_tree(s_[ia3+1:ia4])

    # Rule composition type
    if '{' in s:
        i3 = s.rfind('{')+1
        i4 = s.rfind('}')
        rtype, rclass = s[i3:i4].split(', ')
        node.rule = rtype

    return node



def find_parens(s, p_open, p_close):
    i1 = s.find(p_open)

    x = 1
    for i, c in enumerate(s[i1+1:]):
        if c == '(':
            x += 1
        elif c == ')':
            x -= 1
        if x == 0:
            i2 = i + (i1 + 1)
            return i1, i2

    print 'ERROR: incorrect open/close parens'
    return None





class ParseNode:

    def __init__(self, word, tag, args, head=None):
        self.word   = word
        self.tag    = tag
        self.args   = args
        self.head   = head 
        self.rule   = None
        self.child1 = None
        self.child2 = None
        
    def __str__(self):
        if self.is_leaf():
            nodestr = ' '.join(['<L', self.tag, self.word]) + '> '
        else:
            nodestr = ' '.join(['<T', self.tag, str(self.head), str(self.args)]) + '> '

        if self.args == 0:
            children = ''
        elif self.args == 1:
            children = str(self.child1)
        elif self.args == 2:
            children = str(self.child1) + ' ' + str(self.child2)

        if self.rule is None:
            rulestr = ''
        else:
            rulestr = ' {' + self.rule + '}'

        return '(' + nodestr + children + rulestr + ')'

    def is_leaf(self):
        return False if self.word is None else True





# ------------------------------------------------------------------------

if __name__ == '__main__':
s = "(<T S[dcl] 1 2> (<L NP LS LS i NP>) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBP VBP am (S[dcl]\NP)/NP>) (<T NP 1 2> (<L NP/N DT DT a NP/N>) (<L N NN NN dog N>) {FA, OTHER}) {FA, OTHER}) {BA, OTHER})"
root = parse_tree(s)
    print root


