//! Python source analysis for the code metrics: a CPython-shaped syntax tree (node types, child order and
//! field names as Python's `ast` module, built from `rustpython-parser`) and a tokenizer following Python
//! 3.12's `tokenize` (token kinds, `NL` / `NEWLINE`, indentation and positions), which radon's raw metrics
//! and CodeBLEU's comment stripping depend on.

use rustpython_parser::ast::{self, Constant, Expr, ExprContext, Pattern, Stmt};
use rustpython_parser::Parse;

/// Expression context of `Name`, `Attribute`, `Subscript`, `Starred`, `List` and `Tuple` nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ctx {
    None,
    Load,
    Store,
    Del,
}

/// One AST node: its type name, operator (for `BinOp`, `UnaryOp`, `BoolOp`, `AugAssign`; comparison
/// operators of a `Compare` in `ops`), its child nodes in `ast.iter_child_nodes` order, and the field each
/// child range belongs to.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub tag: &'static str,
    pub op: Option<&'static str>,
    pub ops: Vec<&'static str>,
    pub kids: Vec<Node>,
    pub fields: Vec<(&'static str, usize, usize)>,
    /// `Name.id`, `Attribute.attr`, `arg.arg`, function / class / handler name.
    pub name: Option<String>,
    pub ctx: Ctx,
    /// Constant value as a key with Python's equality (1 == 1.0 == True).
    pub constant: Option<String>,
}

impl Node {
    fn new(tag: &'static str) -> Node {
        Node {
            tag,
            op: None,
            ops: vec![],
            kids: vec![],
            fields: vec![],
            name: None,
            ctx: Ctx::None,
            constant: None,
        }
    }

    fn many(mut self, field: &'static str, nodes: Vec<Node>) -> Node {
        let s = self.kids.len();
        self.kids.extend(nodes);
        self.fields.push((field, s, self.kids.len()));
        self
    }

    fn one(self, field: &'static str, node: Option<Node>) -> Node {
        self.many(field, node.into_iter().collect())
    }

    fn named(mut self, n: impl Into<String>) -> Node {
        self.name = Some(n.into());
        self
    }

    /// The children in field `f` (empty when the node has no such field).
    pub fn field(&self, f: &str) -> &[Node] {
        self.fields
            .iter()
            .find(|x| x.0 == f)
            .map_or(&[], |&(_, a, b)| &self.kids[a..b])
    }

    /// Whether the node has the field (as Python's `hasattr`), even if empty.
    pub fn has(&self, f: &str) -> bool {
        self.fields.iter().any(|x| x.0 == f)
    }

    /// Nodes in `ast.walk` order (breadth first).
    pub fn walk(&self) -> Vec<&Node> {
        let mut out = vec![self];
        let mut i = 0;
        while i < out.len() {
            let n = out[i];
            out.extend(n.kids.iter());
            i += 1;
        }
        out
    }
}

fn stmts(v: &[Stmt]) -> Vec<Node> {
    v.iter().map(stmt).collect()
}

fn exprs(v: &[Expr]) -> Vec<Node> {
    v.iter().map(expr).collect()
}

fn opt(e: &Option<Box<Expr>>) -> Option<Node> {
    e.as_deref().map(expr)
}

fn type_params(v: &[ast::TypeParam]) -> Vec<Node> {
    v.iter()
        .map(|t| match t {
            ast::TypeParam::TypeVar(x) => Node::new("TypeVar")
                .named(x.name.as_str())
                .one("bound", opt(&x.bound)),
            ast::TypeParam::ParamSpec(x) => Node::new("ParamSpec").named(x.name.as_str()),
            ast::TypeParam::TypeVarTuple(x) => Node::new("TypeVarTuple").named(x.name.as_str()),
        })
        .collect()
}

fn arg(a: &ast::Arg) -> Node {
    Node::new("arg")
        .named(a.arg.as_str())
        .one("annotation", a.annotation.as_deref().map(expr))
}

fn arguments(a: &ast::Arguments) -> Node {
    let defaults: Vec<Node> = a
        .posonlyargs
        .iter()
        .chain(&a.args)
        .filter_map(|x| x.default.as_deref().map(expr))
        .collect();
    let kw_defaults: Vec<Node> = a
        .kwonlyargs
        .iter()
        .filter_map(|x| x.default.as_deref().map(expr))
        .collect();
    Node::new("arguments")
        .many(
            "posonlyargs",
            a.posonlyargs.iter().map(|x| arg(&x.def)).collect(),
        )
        .many("args", a.args.iter().map(|x| arg(&x.def)).collect())
        .one("vararg", a.vararg.as_deref().map(arg))
        .many(
            "kwonlyargs",
            a.kwonlyargs.iter().map(|x| arg(&x.def)).collect(),
        )
        .many("kw_defaults", kw_defaults)
        .one("kwarg", a.kwarg.as_deref().map(arg))
        .many("defaults", defaults)
}

fn keywords(v: &[ast::Keyword]) -> Vec<Node> {
    v.iter()
        .map(|k| Node::new("keyword").one("value", Some(expr(&k.value))))
        .collect()
}

fn handlers(v: &[ast::ExceptHandler]) -> Vec<Node> {
    v.iter()
        .map(|ast::ExceptHandler::ExceptHandler(h)| {
            let mut n = Node::new("ExceptHandler")
                .one("type", opt(&h.type_))
                .many("body", stmts(&h.body));
            n.name = h.name.as_ref().map(|x| x.to_string());
            n
        })
        .collect()
}

fn comprehensions(v: &[ast::Comprehension]) -> Vec<Node> {
    v.iter()
        .map(|c| {
            Node::new("comprehension")
                .one("target", Some(expr(&c.target)))
                .one("iter", Some(expr(&c.iter)))
                .many("ifs", exprs(&c.ifs))
        })
        .collect()
}

fn pattern(p: &Pattern) -> Node {
    match p {
        Pattern::MatchValue(x) => Node::new("MatchValue").one("value", Some(expr(&x.value))),
        Pattern::MatchSingleton(_) => Node::new("MatchSingleton"),
        Pattern::MatchSequence(x) => {
            Node::new("MatchSequence").many("patterns", x.patterns.iter().map(pattern).collect())
        }
        Pattern::MatchMapping(x) => Node::new("MatchMapping")
            .many("keys", exprs(&x.keys))
            .many("patterns", x.patterns.iter().map(pattern).collect()),
        Pattern::MatchClass(x) => Node::new("MatchClass")
            .one("cls", Some(expr(&x.cls)))
            .many("patterns", x.patterns.iter().map(pattern).collect())
            .many("kwd_patterns", x.kwd_patterns.iter().map(pattern).collect()),
        Pattern::MatchStar(_) => Node::new("MatchStar"),
        Pattern::MatchAs(x) => {
            Node::new("MatchAs").one("pattern", x.pattern.as_deref().map(pattern))
        }
        Pattern::MatchOr(x) => {
            Node::new("MatchOr").many("patterns", x.patterns.iter().map(pattern).collect())
        }
    }
}

fn with_items(v: &[ast::WithItem]) -> Vec<Node> {
    v.iter()
        .map(|w| {
            Node::new("withitem")
                .one("context_expr", Some(expr(&w.context_expr)))
                .one("optional_vars", opt(&w.optional_vars))
        })
        .collect()
}

fn op_name(o: &ast::Operator) -> &'static str {
    use ast::Operator::*;
    match o {
        Add => "Add",
        Sub => "Sub",
        Mult => "Mult",
        MatMult => "MatMult",
        Div => "Div",
        Mod => "Mod",
        Pow => "Pow",
        LShift => "LShift",
        RShift => "RShift",
        BitOr => "BitOr",
        BitXor => "BitXor",
        BitAnd => "BitAnd",
        FloorDiv => "FloorDiv",
    }
}

fn stmt(s: &Stmt) -> Node {
    match s {
        Stmt::FunctionDef(f) => Node::new("FunctionDef")
            .named(f.name.as_str())
            .one("args", Some(arguments(&f.args)))
            .many("body", stmts(&f.body))
            .many("decorator_list", exprs(&f.decorator_list))
            .one("returns", opt(&f.returns))
            .many("type_params", type_params(&f.type_params)),
        Stmt::AsyncFunctionDef(f) => Node::new("AsyncFunctionDef")
            .named(f.name.as_str())
            .one("args", Some(arguments(&f.args)))
            .many("body", stmts(&f.body))
            .many("decorator_list", exprs(&f.decorator_list))
            .one("returns", opt(&f.returns))
            .many("type_params", type_params(&f.type_params)),
        Stmt::ClassDef(c) => Node::new("ClassDef")
            .named(c.name.as_str())
            .many("bases", exprs(&c.bases))
            .many("keywords", keywords(&c.keywords))
            .many("body", stmts(&c.body))
            .many("decorator_list", exprs(&c.decorator_list))
            .many("type_params", type_params(&c.type_params)),
        Stmt::Return(x) => Node::new("Return").one("value", opt(&x.value)),
        Stmt::Delete(x) => Node::new("Delete").many("targets", exprs(&x.targets)),
        Stmt::Assign(x) => Node::new("Assign")
            .many("targets", exprs(&x.targets))
            .one("value", Some(expr(&x.value))),
        Stmt::TypeAlias(x) => Node::new("TypeAlias")
            .one("name", Some(expr(&x.name)))
            .many("type_params", type_params(&x.type_params))
            .one("value", Some(expr(&x.value))),
        Stmt::AugAssign(x) => {
            let mut n = Node::new("AugAssign")
                .one("target", Some(expr(&x.target)))
                .one("value", Some(expr(&x.value)));
            n.op = Some(op_name(&x.op));
            n
        }
        Stmt::AnnAssign(x) => Node::new("AnnAssign")
            .one("target", Some(expr(&x.target)))
            .one("annotation", Some(expr(&x.annotation)))
            .one("value", opt(&x.value)),
        Stmt::For(x) => Node::new("For")
            .one("target", Some(expr(&x.target)))
            .one("iter", Some(expr(&x.iter)))
            .many("body", stmts(&x.body))
            .many("orelse", stmts(&x.orelse)),
        Stmt::AsyncFor(x) => Node::new("AsyncFor")
            .one("target", Some(expr(&x.target)))
            .one("iter", Some(expr(&x.iter)))
            .many("body", stmts(&x.body))
            .many("orelse", stmts(&x.orelse)),
        Stmt::While(x) => Node::new("While")
            .one("test", Some(expr(&x.test)))
            .many("body", stmts(&x.body))
            .many("orelse", stmts(&x.orelse)),
        Stmt::If(x) => Node::new("If")
            .one("test", Some(expr(&x.test)))
            .many("body", stmts(&x.body))
            .many("orelse", stmts(&x.orelse)),
        Stmt::With(x) => Node::new("With")
            .many("items", with_items(&x.items))
            .many("body", stmts(&x.body)),
        Stmt::AsyncWith(x) => Node::new("AsyncWith")
            .many("items", with_items(&x.items))
            .many("body", stmts(&x.body)),
        Stmt::Match(x) => Node::new("Match")
            .one("subject", Some(expr(&x.subject)))
            .many(
                "cases",
                x.cases
                    .iter()
                    .map(|c| {
                        Node::new("match_case")
                            .one("pattern", Some(pattern(&c.pattern)))
                            .one("guard", opt(&c.guard))
                            .many("body", stmts(&c.body))
                    })
                    .collect(),
            ),
        Stmt::Raise(x) => Node::new("Raise")
            .one("exc", opt(&x.exc))
            .one("cause", opt(&x.cause)),
        Stmt::Try(x) => Node::new("Try")
            .many("body", stmts(&x.body))
            .many("handlers", handlers(&x.handlers))
            .many("orelse", stmts(&x.orelse))
            .many("finalbody", stmts(&x.finalbody)),
        Stmt::TryStar(x) => Node::new("TryStar")
            .many("body", stmts(&x.body))
            .many("handlers", handlers(&x.handlers))
            .many("orelse", stmts(&x.orelse))
            .many("finalbody", stmts(&x.finalbody)),
        Stmt::Assert(x) => Node::new("Assert")
            .one("test", Some(expr(&x.test)))
            .one("msg", opt(&x.msg)),
        Stmt::Import(x) => Node::new("Import").many(
            "names",
            x.names.iter().map(|_| Node::new("alias")).collect(),
        ),
        Stmt::ImportFrom(x) => Node::new("ImportFrom").many(
            "names",
            x.names.iter().map(|_| Node::new("alias")).collect(),
        ),
        Stmt::Global(_) => Node::new("Global"),
        Stmt::Nonlocal(_) => Node::new("Nonlocal"),
        Stmt::Expr(x) => Node::new("Expr").one("value", Some(expr(&x.value))),
        Stmt::Pass(_) => Node::new("Pass"),
        Stmt::Break(_) => Node::new("Break"),
        Stmt::Continue(_) => Node::new("Continue"),
    }
}

fn ctx(c: &ExprContext) -> Ctx {
    match c {
        ExprContext::Load => Ctx::Load,
        ExprContext::Store => Ctx::Store,
        ExprContext::Del => Ctx::Del,
    }
}

fn float_key(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 {
        format!("n:{f:.0}")
    } else {
        format!("n:{f:?}")
    }
}

/// A key for a constant with Python's `==` / hash semantics among literals.
fn constant_key(c: &Constant) -> String {
    match c {
        Constant::None => "None".into(),
        Constant::Ellipsis => "Ellipsis".into(),
        Constant::Bool(b) => format!("n:{}", u8::from(*b)),
        Constant::Int(i) => format!("n:{i}"),
        Constant::Float(f) => float_key(*f),
        Constant::Complex { real, imag } => {
            if *imag == 0.0 {
                float_key(*real)
            } else {
                format!("c:{real:?},{imag:?}")
            }
        }
        Constant::Str(s) => format!("s:{s}"),
        Constant::Bytes(b) => format!("b:{b:?}"),
        Constant::Tuple(t) => format!(
            "t:({})",
            t.iter().map(constant_key).collect::<Vec<_>>().join(",")
        ),
    }
}

fn expr(e: &Expr) -> Node {
    match e {
        Expr::BoolOp(x) => {
            let mut n = Node::new("BoolOp").many("values", exprs(&x.values));
            n.op = Some(match x.op {
                ast::BoolOp::And => "And",
                ast::BoolOp::Or => "Or",
            });
            n
        }
        Expr::NamedExpr(x) => Node::new("NamedExpr")
            .one("target", Some(expr(&x.target)))
            .one("value", Some(expr(&x.value))),
        Expr::BinOp(x) => {
            let mut n = Node::new("BinOp")
                .one("left", Some(expr(&x.left)))
                .one("right", Some(expr(&x.right)));
            n.op = Some(op_name(&x.op));
            n
        }
        Expr::UnaryOp(x) => {
            let mut n = Node::new("UnaryOp").one("operand", Some(expr(&x.operand)));
            n.op = Some(match x.op {
                ast::UnaryOp::Invert => "Invert",
                ast::UnaryOp::Not => "Not",
                ast::UnaryOp::UAdd => "UAdd",
                ast::UnaryOp::USub => "USub",
            });
            n
        }
        Expr::Lambda(x) => Node::new("Lambda")
            .one("args", Some(arguments(&x.args)))
            .one("body", Some(expr(&x.body))),
        Expr::IfExp(x) => Node::new("IfExp")
            .one("test", Some(expr(&x.test)))
            .one("body", Some(expr(&x.body)))
            .one("orelse", Some(expr(&x.orelse))),
        Expr::Dict(x) => Node::new("Dict")
            .many("keys", x.keys.iter().flatten().map(expr).collect())
            .many("values", exprs(&x.values)),
        Expr::Set(x) => Node::new("Set").many("elts", exprs(&x.elts)),
        Expr::ListComp(x) => Node::new("ListComp")
            .one("elt", Some(expr(&x.elt)))
            .many("generators", comprehensions(&x.generators)),
        Expr::SetComp(x) => Node::new("SetComp")
            .one("elt", Some(expr(&x.elt)))
            .many("generators", comprehensions(&x.generators)),
        Expr::DictComp(x) => Node::new("DictComp")
            .one("key", Some(expr(&x.key)))
            .one("value", Some(expr(&x.value)))
            .many("generators", comprehensions(&x.generators)),
        Expr::GeneratorExp(x) => Node::new("GeneratorExp")
            .one("elt", Some(expr(&x.elt)))
            .many("generators", comprehensions(&x.generators)),
        Expr::Await(x) => Node::new("Await").one("value", Some(expr(&x.value))),
        Expr::Yield(x) => Node::new("Yield").one("value", opt(&x.value)),
        Expr::YieldFrom(x) => Node::new("YieldFrom").one("value", Some(expr(&x.value))),
        Expr::Compare(x) => {
            let mut n = Node::new("Compare")
                .one("left", Some(expr(&x.left)))
                .many("comparators", exprs(&x.comparators));
            n.ops = x
                .ops
                .iter()
                .map(|o| {
                    use ast::CmpOp::*;
                    match o {
                        Eq => "Eq",
                        NotEq => "NotEq",
                        Lt => "Lt",
                        LtE => "LtE",
                        Gt => "Gt",
                        GtE => "GtE",
                        Is => "Is",
                        IsNot => "IsNot",
                        In => "In",
                        NotIn => "NotIn",
                    }
                })
                .collect();
            n
        }
        Expr::Call(x) => Node::new("Call")
            .one("func", Some(expr(&x.func)))
            .many("args", exprs(&x.args))
            .many("keywords", keywords(&x.keywords)),
        Expr::FormattedValue(x) => Node::new("FormattedValue")
            .one("value", Some(expr(&x.value)))
            .one("format_spec", opt(&x.format_spec)),
        Expr::JoinedStr(x) => Node::new("JoinedStr").many("values", exprs(&x.values)),
        Expr::Constant(x) => {
            let mut n = Node::new("Constant");
            n.constant = Some(constant_key(&x.value));
            n
        }
        Expr::Attribute(x) => {
            let mut n = Node::new("Attribute")
                .one("value", Some(expr(&x.value)))
                .named(x.attr.as_str());
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::Subscript(x) => {
            let mut n = Node::new("Subscript")
                .one("value", Some(expr(&x.value)))
                .one("slice", Some(expr(&x.slice)));
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::Starred(x) => {
            let mut n = Node::new("Starred").one("value", Some(expr(&x.value)));
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::Name(x) => {
            let mut n = Node::new("Name").named(x.id.as_str());
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::List(x) => {
            let mut n = Node::new("List").many("elts", exprs(&x.elts));
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::Tuple(x) => {
            let mut n = Node::new("Tuple").many("elts", exprs(&x.elts));
            n.ctx = ctx(&x.ctx);
            n
        }
        Expr::Slice(x) => Node::new("Slice")
            .one("lower", opt(&x.lower))
            .one("upper", opt(&x.upper))
            .one("step", opt(&x.step)),
    }
}

/// Parses Python source into a `Module` node; `None` when it does not parse.
pub(crate) fn parse(src: &str) -> Option<Node> {
    let suite = ast::Suite::parse(src, "<generated>").ok()?;
    Some(
        Node::new("Module")
            .many("body", stmts(&suite))
            .many("type_ignores", vec![]),
    )
}

// ------------------------------------------------------------------------------------------- tokenizer

/// Token kinds of Python's `tokenize` (f-strings as one `FString` token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tok {
    Name,
    Number,
    String,
    FString,
    Op,
    Comment,
    Nl,
    Newline,
    Indent,
    Dedent,
    EndMarker,
}

#[derive(Debug, Clone)]
pub(crate) struct Token {
    pub kind: Tok,
    pub text: String,
    pub start: (usize, usize),
    pub end: (usize, usize),
}

const OPS3: [&str; 5] = ["**=", "//=", ">>=", "<<=", "..."];
const OPS2: [&str; 21] = [
    "->", ":=", "**", "//", ">>", "<<", "<=", ">=", "==", "!=", "+=", "-=", "*=", "/=", "%=", "&=",
    "|=", "^=", "@=", "<>", "!=",
];

fn is_id_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_id_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Length (in chars) of a string prefix + opening quote at `i`, with (is_fstring, quote char, triple).
fn string_start(c: &[char], i: usize) -> Option<(usize, bool, char, bool)> {
    let mut j = i;
    while j < c.len() && j - i < 2 && "rRbBuUfF".contains(c[j]) {
        j += 1;
    }
    if j >= c.len() || (c[j] != '\'' && c[j] != '"') {
        return None;
    }
    let prefix: String = c[i..j].iter().collect::<String>().to_lowercase();
    if !["", "r", "u", "b", "br", "rb", "f", "fr", "rf"].contains(&prefix.as_str()) {
        return None;
    }
    let q = c[j];
    let triple = j + 2 < c.len() && c[j + 1] == q && c[j + 2] == q;
    Some((
        j - i + if triple { 3 } else { 1 },
        prefix.contains('f'),
        q,
        triple,
    ))
}

/// Colons that Python's tokenizer reports as `OP ':'` inside an f-string body (expression parts and the
/// first format-spec colon of each replacement field).
fn fstring_colons(body: &[char]) -> usize {
    fn field(b: &[char], mut i: usize, n: &mut usize) -> usize {
        // `i` is just after '{'; returns the index after the matching '}'.
        let mut depth = 0i32;
        let mut spec = false;
        while i < b.len() {
            let ch = b[i];
            if spec {
                match ch {
                    '{' => i = field(b, i + 1, n),
                    '}' => return i + 1,
                    _ => i += 1,
                }
                continue;
            }
            match ch {
                '\'' | '"' => {
                    let q = ch;
                    i += 1;
                    while i < b.len() && b[i] != q {
                        i += 1;
                    }
                    i += 1;
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    i += 1;
                }
                ')' | ']' => {
                    depth -= 1;
                    i += 1;
                }
                '}' if depth == 0 => return i + 1,
                '}' => {
                    depth -= 1;
                    i += 1;
                }
                ':' if i + 1 < b.len() && b[i + 1] == '=' && depth > 0 => i += 2,
                ':' => {
                    *n += 1;
                    if depth == 0 {
                        spec = true;
                    }
                    i += 1;
                }
                _ => i += 1,
            }
        }
        i
    }
    let mut n = 0;
    let mut i = 0;
    while i < body.len() {
        if body[i] == '{' {
            if i + 1 < body.len() && body[i + 1] == '{' {
                i += 2;
            } else {
                i = field(body, i + 1, &mut n);
            }
        } else {
            i += 1;
        }
    }
    n
}

/// Tokenizes `src` as Python 3.12's `tokenize.generate_tokens`; `Err` for what it rejects (unterminated
/// strings, EOF inside brackets or after a line continuation, inconsistent dedents, unknown characters).
pub(crate) fn tokenize(src: &str) -> Result<Vec<Token>, ()> {
    let mut lines: Vec<Vec<char>> = Vec::new();
    {
        let mut cur = Vec::new();
        for ch in src.chars() {
            cur.push(ch);
            if ch == '\n' {
                lines.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
    }
    let mut out: Vec<Token> = Vec::new();
    let mut indents = vec![0usize];
    let (mut paren, mut continued) = (0i32, false);
    // A pending multi-line string: (text so far, start, quote, triple, is_f).
    let mut cont: Option<(String, (usize, usize), char, bool, bool)> = None;
    let push = |out: &mut Vec<Token>, kind, text: String, start, end| {
        out.push(Token {
            kind,
            text,
            start,
            end,
        })
    };
    let mut lnum = 0;
    for line in &lines {
        lnum += 1;
        let max = line.len();
        let mut pos = 0;
        if let Some((mut text, start, q, triple, is_f)) = cont.take() {
            // Find the closing quote.
            let mut i = 0;
            let mut found = None;
            while i < max {
                if line[i] == '\\' {
                    i += 2;
                    continue;
                }
                if line[i] == q
                    && (!triple || (i + 2 < max && line[i + 1] == q && line[i + 2] == q))
                {
                    found = Some(i + if triple { 3 } else { 1 });
                    break;
                }
                if !triple && line[i] == '\n' {
                    break;
                }
                i += 1;
            }
            match found {
                Some(e) => {
                    text.extend(&line[..e]);
                    let kind = if is_f { Tok::FString } else { Tok::String };
                    if is_f {
                        let body: Vec<char> = text.chars().collect();
                        for _ in 0..fstring_colons(&body) {
                            push(&mut out, Tok::Op, ":".into(), start, start);
                        }
                    }
                    push(&mut out, kind, text, start, (lnum, e));
                    pos = e;
                }
                None => {
                    if !triple && !(max >= 2 && line[max - 1] == '\n' && line[max - 2] == '\\') {
                        return Err(());
                    }
                    text.extend(line.iter());
                    cont = Some((text, start, q, triple, is_f));
                    continue;
                }
            }
        } else if paren == 0 && !continued {
            let mut column = 0;
            while pos < max {
                match line[pos] {
                    ' ' => column += 1,
                    '\t' => column = (column / 8 + 1) * 8,
                    '\x0c' => column = 0,
                    _ => break,
                }
                pos += 1;
            }
            if pos == max {
                // Whitespace only up to EOF (no newline): nothing more on this line.
                break;
            }
            if matches!(line[pos], '#' | '\r' | '\n') {
                if line[pos] == '#' {
                    let mut e = pos;
                    while e < max && line[e] != '\r' && line[e] != '\n' {
                        e += 1;
                    }
                    push(
                        &mut out,
                        Tok::Comment,
                        line[pos..e].iter().collect(),
                        (lnum, pos),
                        (lnum, e),
                    );
                    pos = e;
                }
                push(
                    &mut out,
                    Tok::Nl,
                    line[pos..].iter().collect(),
                    (lnum, pos),
                    (lnum, max),
                );
                continue;
            }
            if column > *indents.last().unwrap() {
                indents.push(column);
                push(
                    &mut out,
                    Tok::Indent,
                    line[..pos].iter().collect(),
                    (lnum, 0),
                    (lnum, pos),
                );
            }
            while column < *indents.last().unwrap() {
                if !indents.contains(&column) {
                    return Err(());
                }
                indents.pop();
                push(
                    &mut out,
                    Tok::Dedent,
                    String::new(),
                    (lnum, pos),
                    (lnum, pos),
                );
            }
        } else {
            continued = false;
        }
        while pos < max {
            let ch = line[pos];
            if ch == ' ' || ch == '\t' || ch == '\x0c' {
                pos += 1;
                continue;
            }
            let start = (lnum, pos);
            if ch == '#' {
                let mut e = pos;
                while e < max && line[e] != '\r' && line[e] != '\n' {
                    e += 1;
                }
                push(
                    &mut out,
                    Tok::Comment,
                    line[pos..e].iter().collect(),
                    start,
                    (lnum, e),
                );
                pos = e;
                continue;
            }
            if ch == '\r' || ch == '\n' {
                let kind = if paren > 0 { Tok::Nl } else { Tok::Newline };
                push(
                    &mut out,
                    kind,
                    line[pos..].iter().collect(),
                    start,
                    (lnum, max),
                );
                pos = max;
                continue;
            }
            if ch == '\\' {
                let rest: String = line[pos + 1..].iter().collect();
                if rest == "\n" || rest == "\r\n" {
                    continued = true;
                    pos = max;
                    continue;
                }
                return Err(());
            }
            if let Some((plen, is_f, q, triple)) = string_start(line, pos) {
                let mut i = pos + plen;
                let mut end = None;
                while i < max {
                    if line[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if !triple && (line[i] == '\n' || line[i] == '\r') {
                        break;
                    }
                    if line[i] == q
                        && (!triple || (i + 2 < max && line[i + 1] == q && line[i + 2] == q))
                    {
                        end = Some(i + if triple { 3 } else { 1 });
                        break;
                    }
                    i += 1;
                }
                match end {
                    Some(e) => {
                        let text: String = line[pos..e].iter().collect();
                        if is_f {
                            let body: Vec<char> = text.chars().collect();
                            for _ in 0..fstring_colons(&body) {
                                push(&mut out, Tok::Op, ":".into(), start, start);
                            }
                        }
                        push(
                            &mut out,
                            if is_f { Tok::FString } else { Tok::String },
                            text,
                            start,
                            (lnum, e),
                        );
                        pos = e;
                    }
                    None => {
                        let ends_cont = max >= 2 && line[max - 1] == '\n' && line[max - 2] == '\\';
                        if !triple && !ends_cont {
                            return Err(());
                        }
                        cont = Some((line[pos..].iter().collect(), start, q, triple, is_f));
                        pos = max;
                    }
                }
                continue;
            }
            if is_id_start(ch) {
                let mut e = pos + 1;
                while e < max && is_id_char(line[e]) {
                    e += 1;
                }
                push(
                    &mut out,
                    Tok::Name,
                    line[pos..e].iter().collect(),
                    start,
                    (lnum, e),
                );
                pos = e;
                continue;
            }
            if ch.is_ascii_digit() || (ch == '.' && pos + 1 < max && line[pos + 1].is_ascii_digit())
            {
                let mut e = pos + 1;
                while e < max {
                    let c = line[e];
                    let prev = line[e - 1];
                    if c.is_ascii_alphanumeric()
                        || c == '_'
                        || c == '.'
                        || ((c == '+' || c == '-')
                            && (prev == 'e' || prev == 'E')
                            && !line[pos..e].iter().any(|x| *x == 'x' || *x == 'X'))
                    {
                        e += 1;
                    } else {
                        break;
                    }
                }
                push(
                    &mut out,
                    Tok::Number,
                    line[pos..e].iter().collect(),
                    start,
                    (lnum, e),
                );
                pos = e;
                continue;
            }
            let rest: String = line[pos..max.min(pos + 3)].iter().collect();
            let op = OPS3
                .iter()
                .chain(OPS2.iter())
                .find(|o| rest.starts_with(**o))
                .map(|o| o.to_string())
                .or_else(|| {
                    "()[]{}:,;+-*/|&<>=.%`~^@!"
                        .contains(ch)
                        .then(|| ch.to_string())
                });
            let Some(op) = op else { return Err(()) };
            match op.as_str() {
                "(" | "[" | "{" => paren += 1,
                ")" | "]" | "}" => {
                    paren -= 1;
                    if paren < 0 {
                        return Err(());
                    }
                }
                _ => {}
            }
            let e = pos + op.chars().count();
            push(&mut out, Tok::Op, op, start, (lnum, e));
            pos = e;
        }
    }
    if cont.is_some() || paren > 0 || continued {
        return Err(());
    }
    let last = lines.len();
    if let Some(t) = out.last() {
        if !matches!(t.kind, Tok::Newline | Tok::Nl | Tok::Dedent) {
            let col = lines.last().map_or(0, Vec::len);
            push(
                &mut out,
                Tok::Newline,
                String::new(),
                (last, col),
                (last, col + 1),
            );
        }
    }
    let end = (last + 1, 0);
    for _ in 1..indents.len() {
        push(&mut out, Tok::Dedent, String::new(), end, end);
    }
    push(&mut out, Tok::EndMarker, String::new(), end, end);
    Ok(out)
}
