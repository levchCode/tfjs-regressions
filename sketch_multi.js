let x1_vals = [];
let x2_vals = [];
let y_vals = [];

let a, b1, b2, n, m, ttab

let ctx;
let lineX = [0, 13];

let optimizer = tf.train.adamax(0.5)

function preload() {
  dataset = loadTable('data_multi.csv','csv','header')
}

function setup() {
  noCanvas()

  ctx = document.getElementById('scatter');
  
  n = dataset.getRowCount()
  for (var i = 0; i < n; i++) {
    x1_vals.push(dataset.getNum(i, "x1"))
    x2_vals.push(dataset.getNum(i, "x2"))
    y_vals.push(dataset.getNum(i, "y"))
  }

  // x1_min = Math.min(...x1_vals)
  // x1_max = Math.max(...x1_vals)
  // x2_min = Math.min(...x2_vals)
  // x2_max = Math.max(...x2_vals)
  // y_min = Math.min(...y_vals)
  // y_max = Math.max(...y_vals)


  // for (i = 0; i < n; i++) {
  //   x1 = (x1_vals[i] - x1_min)/(x1_max-x1_min)
  //   x2 = (x2_vals[i] - x2_min)/(x2_max-x2_min)
  //   y = (y_vals[i] - y_min)/(y_max-y_min)
  //   x1_vals[i] = x1
  //   x2_vals[i] = x2
  //   y_vals[i] = y
  // }

  
  a = tf.variable(tf.scalar(0));
  b1 = tf.variable(tf.scalar(0));
  b2 = tf.variable(tf.scalar(0));
  m = tf.scalar(2);
  n = tf.scalar(n);
  
}

function dev(x)
{
  return x.sub(x.mean()).square().sum().div(n.sub(tf.scalar(1)))
}

function stddev(x)
{
  return tf.sqrt(dev(x))
}

function corr(x, y)
{
  divz = stddev(x).mul(stddev(y))
  return (x.mul(y).mean().sub(x.mean().mul(y.mean()))).div(divz)
}


function stats(x1, x2, y, y_p)
{
  // document.getElementById("x1x1").innerHTML = corr(x1,x1).dataSync()[0].toFixed(2)
  // document.getElementById("x1x2").innerHTML = corr(x1,x2).dataSync()[0].toFixed(2)
  // document.getElementById("x1y").innerHTML = corr(x1,y).dataSync()[0].toFixed(2)

  // document.getElementById("x2x1").innerHTML = corr(x2,x1).dataSync()[0].toFixed(2)
  // document.getElementById("x2x2").innerHTML = corr(x2,x2).dataSync()[0].toFixed(2)
  // document.getElementById("x2y").innerHTML = corr(x2,y).dataSync()[0].toFixed(2)

  // document.getElementById("yx1").innerHTML = corr(y, x1).dataSync()[0].toFixed(2)
  // document.getElementById("yx2").innerHTML = corr(y, x2).dataSync()[0].toFixed(2)
  // document.getElementById("yy").innerHTML = corr(y, y).dataSync()[0].toFixed(2)

    let elas1 = b1.mul(x1.mean().div(y.mean()))
    let elas2 = b2.mul(x2.mean().div(y.mean()))

    let residual_sq = y.sub(y_p).square().sum().div(n)

    let R = tf.scalar(1).sub(residual_sq.div(dev(y)))

    let Fr = R.div(tf.scalar(1).sub(R)).mul(n.sub(m).sub(tf.scalar(1)))
    let F1 = R.sub(corr(y, x2).square()).div(tf.scalar(1).sub(R)).mul(n.sub(tf.scalar(3)))
    let F2 = R.sub(corr(y, x1).square()).div(tf.scalar(1).sub(R)).mul(n.sub(tf.scalar(3)))

    let tb1 = F1.sqrt().dataSync()[0]
    let tb2 = F2.sqrt().dataSync()[0]


    document.getElementById("e1").innerHTML = "Э<sub>1</sub> = " +  elas1.dataSync()[0].toFixed(5)
    document.getElementById("e2").innerHTML = "Э<sub>2</sub> = " +  elas2.dataSync()[0].toFixed(5)
    document.getElementById("rs").innerHTML = "σ<sub>ост</sub> = " +  residual_sq.dataSync()[0].toFixed(5)
    document.getElementById("R").innerHTML = "R<sup>2</sup><sub>yx<sub>1</sub>x<sub>2</sub></sub> = " + R.dataSync()[0].toFixed(5)
  
    document.getElementById("Fr").innerHTML = "F<sub>факт</sub> = " + Fr.dataSync()[0].toFixed(5)
    document.getElementById("Fx1").innerHTML = "F<sub>x1</sub> = " +  F1.dataSync()[0].toFixed(5)
    document.getElementById("Fx2").innerHTML = "F<sub>x2</sub> = " +  F2.dataSync()[0].toFixed(5)
    document.getElementById("tb1").innerHTML = "t<sub>b1</sub> = " + tb1.toFixed(5)
    document.getElementById("tb2").innerHTML = "t<sub>b2</sub> = " + tb2.toFixed(5)

    if (F1.dataSync()[0] >= parseFloat(document.getElementById("f").value))
    {
      document.getElementById("Fx1").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("Fx1").style.cssText = "color: red"
    }

    if (F2.dataSync()[0] >= parseFloat(document.getElementById("f").value))
    {
      document.getElementById("Fx2").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("Fx2").style.cssText = "color: red"
    }

    if (Fr.dataSync()[0] >= parseFloat(document.getElementById("f").value))
    {
      document.getElementById("Fr").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("Fr").style.cssText = "color: red"
    }
}

function loss(pred, labels) 
{
	return pred.sub(labels).square().mean();
}


function predict(x1, x2) {

  x1 = tf.tensor1d(x1);
  x2 = tf.tensor1d(x2);
  y = tf.tensor1d(y_vals);

  beta1 = b1.mul(stddev(x1).div(stddev(y)))
  beta2 = b2.mul(stddev(x2).div(stddev(y)))

  // y = a + bx;
  let y_p = x1.mul(b1).add(x2.mul(b2)).add(a);
  document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + " + " + b1.dataSync()[0].toFixed(4) + "x1 + " + b2.dataSync()[0].toFixed(4) + "x2"
  document.getElementById("teq").innerHTML = "t<sub>y</sub> = " + beta1.dataSync()[0].toFixed(4) + "t<sub>x1</sub> + " + beta2.dataSync()[0].toFixed(4) + "t<sub>x2</sub>"

  stats(x1, x2, y, y_p)
  return y_p
}



function draw() {

  tf.tidy(() => {
      const y = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x1_vals, x2_vals), y));
  });
  
  console.log(tf.memory().numTensors);
  //noLoop();
}
