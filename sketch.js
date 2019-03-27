let x_vals = [];
let y_vals = [];

let a, b, c, n, mn1, ttab

let ctx;
let lineX = [];

let optimizer = tf.train.adam(0.05)

function preload() {
  dataset = loadTable('data.csv','csv','header')
}

function setup() {
  noCanvas()

  ctx = document.getElementById('scatter');
  
  n = dataset.getRowCount()
  d = []
  for (var i = 0; i < n; i++) {
    x = dataset.getNum(i, "x")
    y = dataset.getNum(i, "y")
    d.push({x:x, y:y})
    x_vals.push(x)
    y_vals.push(y)
  }

  x_min = Math.min(...x_vals)
  x_max = Math.max(...x_vals)
  y_min = Math.min(...y_vals)
  y_max = Math.max(...y_vals)

  d = []
  for (i = 0; i < n; i++) {
    x = (x_vals[i] - x_min)/(x_max-x_min)
    y = (y_vals[i] - y_min)/(y_max-y_min)
    d.push({x:x, y:y})
    x_vals[i] = x + 0.000001
    y_vals[i] = y + 0.000001
  }

  for (i = 0; i < n; i++) {
    lineX.push(i/5)
  }
  
  a = tf.variable(tf.scalar(0));
  b = tf.variable(tf.scalar(0));
  c = tf.variable(tf.scalar(0));
  nm1 = tf.variable(tf.scalar(n-2));
  n = tf.variable(tf.scalar(n));
  
}

function run()
{
  a = tf.variable(tf.scalar(0));
  b = tf.variable(tf.scalar(0));
  c = tf.variable(tf.scalar(0));
}

function stats(x, y, y_p)
{

    let residual_dev = y_p.sub(y).square().sum().div(n)

    let res_dev_t = y_p.sub(y).square().sum().div(nm1)

    //let A = y.sub(y_p).div(y).abs().sum().div(n)

    let mb = tf.sqrt(res_dev_t.div(x.sub(x.mean()).square().sum()))
    let ma = tf.sqrt((res_dev_t.mul(x.square().sum())).div(n.mul(x.sub(x.mean()).square().sum())))

    let ta = a.div(ma).dataSync()
    let tb = b.div(mb).dataSync()

    //document.getElementById("a").innerHTML = "A = " + A.dataSync()[0].toFixed(4) + " %"
    document.getElementById("rsd").innerHTML = "Остаточная дисперсия = " + residual_dev.dataSync()[0].toFixed(4)
    document.getElementById("mb").innerHTML = "m<sub>b</sub> = " + mb.dataSync()[0].toFixed(4)
    document.getElementById("ma").innerHTML = "m<sub>a</sub> = " + ma.dataSync()[0].toFixed(4)
    document.getElementById("tb").innerHTML = "t<sub>b</sub> = " + tb[0].toFixed(4)
    document.getElementById("ta").innerHTML = "t<sub>a</sub> = " + ta[0].toFixed(4)

    if (tb >= parseFloat(document.getElementById("t").value))
    {
      document.getElementById("tb").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("tb").style.cssText = "color: red"
    }

    if (ta >= parseFloat(document.getElementById("t").value))
    {
      document.getElementById("ta").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("ta").style.cssText = "color: red"
    }
}

function loss(pred, labels) 
{
	return pred.sub(labels).square().mean()
}


function predict(x) {
  var e = document.getElementById("sel");
  var model= e.options[e.selectedIndex].innerHTML;

  x = tf.tensor1d(x);
  y = tf.tensor1d(y_vals);

  let y_p; 

  switch(model)
  {
    case "Линейная":
    // y = a + bx;
    y_p = x.mul(b).add(a);
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + " + " + b.dataSync()[0].toFixed(4) + "x"
    break;
    case "Полиноминальная": 
    // y = a + bx + cx^2;
    y_p = x.mul(b).add(x.square().mul(c)).add(a);
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + " + " + b.dataSync()[0].toFixed(4) + "x + " + c.dataSync()[0].toFixed(4) + "x^2"
    break;
    case "Полулогарифмическая": 
    // y = a + b*lnx;
    y_p = b.mul(x.log()).add(a);
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + " + " + b.dataSync()[0].toFixed(4) + "ln(x)"
    break;
    case "Степенная": 
    // y = ax^b;
    y_p = a.mul(x.pow(b));
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + "x^" + b.dataSync()[0].toFixed(4)
    break;
    case "Показательная": 
    // y = ab^x;
    y_p = a.mul(b.pow(x));
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + "*" + b.dataSync()[0].toFixed(4) + "^x"
    break;
    case "Гиперболическая": 
    // y = a+b/x;
    y_p = b.div(x).add(a);
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + "+(" + b.dataSync()[0].toFixed(4) + "/x)"
    break;
    case "Квадратный корень": 
    // y = a+b*sqrt(x);
    y_p = b.mul(x.sqrt()).add(a);
    document.getElementById("eq").innerHTML = "y = " + a.dataSync()[0].toFixed(4) + "+" + b.dataSync()[0].toFixed(4) + "sqrt(x)"
    break;
  }

  stats(x, y, y_p)
  return y_p
}



function draw() {
  
  tf.tidy(() => {
      const y = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), y));
  });
  
  const ys = tf.tidy(() => predict(lineX));

  reg_pred = []
  for (var i = 0; i < lineX.length; i++) {
    reg_pred.push({x:lineX[i], y:ys.dataSync()[i]})
  }
  
	new Chart(ctx, {
    type: 'scatter',
    options: {
      responsive:false,
      animation: false
    },
    data: {
        datasets: [
          {
            label:"Данные",
            backgroundColor: 'rgb(255, 99, 132)',
            data: d
          },
          {
            label:"Регрессия",
            type:"line",
            backgroundColor: 'rgb(0, 0, 0)',
            borderColor: 'black',
            showLine: true,
            fill: false,
            data:reg_pred,
          }
      ]
    }
  }
);
ys.dispose();

  console.log(tf.memory().numTensors);
}
