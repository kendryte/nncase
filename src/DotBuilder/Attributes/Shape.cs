namespace DotBuilder.Attributes
{
    // http://www.graphviz.org/content/node-shapes
    public class Shape : Attribute, INodeAttribute
    {
        public Shape(string value) : base(value)
        {
        }

        public static Shape Assembly => new Shape("assembly");
        public static Shape Box => new Shape("box");
        public static Shape Box3D => new Shape("box3d");
        public static Shape Cds => new Shape("cds");
        public static Shape Circle => new Shape("circle");
        public static Shape Component => new Shape("component");
        public static Shape Cylinder => new Shape("cylinder");
        public static Shape Diamond => new Shape("diamond");
        public static Shape Doublecircle => new Shape("doublecircle");
        public static Shape Doubleoctagon => new Shape("doubleoctagon");
        public static Shape Egg => new Shape("egg");
        public static Shape Ellipse => new Shape("ellipse");
        public static Shape Fivepoverhang => new Shape("fivepoverhang");
        public static Shape Folder => new Shape("folder");
        public static Shape Hexagon => new Shape("hexagon");
        public static Shape House => new Shape("house");
        public static Shape Insulator => new Shape("insulator");
        public static Shape Invhouse => new Shape("invhouse");
        public static Shape Invtrapezium => new Shape("invtrapezium");
        public static Shape Invtriangle => new Shape("invtriangle");
        public static Shape Larrow => new Shape("larrow");
        public static Shape Lpromoter => new Shape("lpromoter");
        public static Shape Mcircle => new Shape("Mcircle");
        public static Shape Mdiamond => new Shape("Mdiamond");
        public static Shape Msquare => new Shape("Msquare");
        public static Shape None => new Shape("none");
        public static Shape Note => new Shape("note");
        public static Shape Noverhang => new Shape("noverhang");
        public static Shape Octagon => new Shape("octagon");
        public static Shape Oval => new Shape("oval");
        public static Shape Parallelogram => new Shape("parallelogram");
        public static Shape Pentagon => new Shape("pentagon");
        public static Shape Plain => new Shape("plain");
        public static Shape Plaintext => new Shape("plaintext");
        public static Shape Point => new Shape("point");
        public static Shape Polygon => new Shape("polygon");
        public static Shape Primersite => new Shape("primersite");
        public static Shape Promoter => new Shape("promoter");
        public static Shape Proteasesite => new Shape("proteasesite");
        public static Shape Proteinstab => new Shape("proteinstab");
        public static Shape Rarrow => new Shape("rarrow");
        public static Shape Record => new Shape("record");
        public static Shape Rect => new Shape("rect");
        public static Shape Rectangle => new Shape("rectangle");
        public static Shape Restrictionsite => new Shape("restrictionsite");
        public static Shape Ribosite => new Shape("ribosite");
        public static Shape Rnastab => new Shape("rnastab");
        public static Shape Rpromoter => new Shape("rpromoter");
        public static Shape Septagon => new Shape("septagon");
        public static Shape Signature => new Shape("signature");
        public static Shape Square => new Shape("square");
        public static Shape Star => new Shape("star");
        public static Shape Tab => new Shape("tab");
        public static Shape Terminator => new Shape("terminator");
        public static Shape Threepoverhang => new Shape("threepoverhang");
        public static Shape Trapezium => new Shape("trapezium");
        public static Shape Triangle => new Shape("triangle");
        public static Shape Tripleoctagon => new Shape("tripleoctagon");
        public static Shape Underline => new Shape("underline");
        public static Shape Utr => new Shape("utr");
    }
}