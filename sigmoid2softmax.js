var dl = require('deeplearn');
var math = new dl.NDArrayMathCPU();
const session = new dl.Session(g, math);
var target = [dl.Array1D.new([1,2,3,4,5])];

//建立图
var g = new dl.Graph();

var X = g.placeholder('X',[5]);
var sf1 = g.softmax(X);

var _t1 = g.multiply(X,g.constant(-1));
var _tsg = g.sigmoid(_t1);
var _t2 = g.divide(g.constant(1),_tsg);
var _t3 = g.subtract(_t2,g.constant(1));
var sf2 = g.divide(_t3,g.reduceSum(_t3));


const shuffledInputProviderBuilder = new dl.InCPUMemoryShuffledInputProviderBuilder([target]);
    
const inputX = shuffledInputProviderBuilder.getInputProviders()[0];
    


const feedEntries = [{
            tensor: X,
            data: inputX
        }
    ];


console.log(session.eval(sf1,feedEntries ).getValues());


//console.log(session.eval(_t1,feedEntries ).getValues());
//console.log(session.eval(_tsg,feedEntries ).getValues());
//console.log(session.eval(_t2,feedEntries ).getValues());
//console.log(session.eval(_t3,feedEntries ).getValues());
console.log(session.eval(sf2,feedEntries ).getValues());

