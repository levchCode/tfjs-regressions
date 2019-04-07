let x_vals = [];
let y_vals = [];

let a, b, c, n, mn1, ttab

let ctx;
let lineX = [];

let optimizer = tf.train.adamax(0.5)

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

  // x_min = Math.min(...x_vals)
  // x_max = Math.max(...x_vals)
  // y_min = Math.min(...y_vals)
  // y_max = Math.max(...y_vals)

  // d = []
  // for (i = 0; i < n; i++) {
  //   x = (x_vals[i] - x_min)/(x_max-x_min)
  //   y = (y_vals[i] - y_min)/(y_max-y_min)
  //   d.push({x:x, y:y})
  //   x_vals[i] = x + 0.000001
  //   y_vals[i] = y + 0.000001
  // }

  for (i = 0; i < n; i++) {
    lineX.push(i*3)
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

function dev2(x) {
  return x.sub(x.mean()).square().sum().div(n)
}

function stats(x, y, y_p)
{

    let residual_dev = y.sub(y_p).square().sum().div(n)

    let r2 = tf.scalar(1).sub(residual_dev.div(dev2(y))) 

    let A = y.sub(y_p).div(y).abs().sum().div(n).mul(tf.scalar(100))

    let F_l = y_p.sub(y.mean()).square().sum().div(y.sub(y_p).square().sum()).mul(n.sub(tf.scalar(2))).dataSync()[0]
    let F_n = r2.div(tf.scalar(1).sub(r2)).mul(n.sub(tf.scalar(2))).dataSync()[0]

    let s_res = y.sub(y_p).square().sum().div(n.sub(tf.scalar(2)))

    let mb = s_res.div(dev2(x).sqrt().mul(n.sqrt()))
    let ma = s_res.mul(x.square().sum().sqrt().div(dev2(x).sqrt().mul(n)))
    let mr = tf.scalar(1).sub(r2).div(n.sub(tf.scalar(2)))

    let ta = a.div(ma).dataSync()
    let tb = b.div(mb).dataSync()
    let tr = r2.div(mr).dataSync()

    let a_l = a.sub(tf.scalar(parseFloat(document.getElementById("t").value)).mul(ma))
    let a_h = a.add(tf.scalar(parseFloat(document.getElementById("t").value)).mul(ma))

    let b_l = b.sub(tf.scalar(parseFloat(document.getElementById("t").value)).mul(mb))
    let b_h = b.add(tf.scalar(parseFloat(document.getElementById("t").value)).mul(mb))

    document.getElementById("rsd").innerHTML = "σ<sub>ост</sub> = " + residual_dev.dataSync()[0].toFixed(4)
    document.getElementById("sr").innerHTML = "S<sub>ост</sub> = " + s_res.dataSync()[0].toFixed(4)
    document.getElementById("r2").innerHTML = "r<sup>2</sup> = " + r2.dataSync()[0].toFixed(4)
    document.getElementById("a").innerHTML = "A = " + A.dataSync()[0].toFixed(4)
    document.getElementById("F").innerHTML = "F = " + F_l.toFixed(4)
    document.getElementById("F2").innerHTML = "F = " + F_n.toFixed(4)
    document.getElementById("mb").innerHTML = "m<sub>b</sub> = " + mb.dataSync()[0].toFixed(4)
    document.getElementById("ma").innerHTML = "m<sub>a</sub> = " + ma.dataSync()[0].toFixed(4)
    document.getElementById("mr").innerHTML = "m<sub>r</sub> = " + mr.dataSync()[0].toFixed(4)
    document.getElementById("tb").innerHTML = "t<sub>b</sub> = " + tb[0].toFixed(4)
    document.getElementById("ta").innerHTML = "t<sub>a</sub> = " + ta[0].toFixed(4)
    document.getElementById("tr").innerHTML = "t<sub>r</sub> = " + tr[0].toFixed(4)
    document.getElementById("ai").innerHTML = "Доверительный интервал (a) = [" + a_l.dataSync()[0].toFixed(4) + "; " + a_h.dataSync()[0].toFixed(4) + "]"
    document.getElementById("bi").innerHTML = "Доверительный интервал (b) = [" + b_l.dataSync()[0].toFixed(4) + "; " + b_h.dataSync()[0].toFixed(4) + "]"

    if ( F_l >= parseFloat(document.getElementById("f").value))
    {
      document.getElementById("F").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("F").style.cssText = "color: red"
    }

    if (F_n >= parseFloat(document.getElementById("f").value))
    {
      document.getElementById("F2").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("F2").style.cssText = "color: red"
    }

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

    if (tr >= parseFloat(document.getElementById("t").value))
    {
      document.getElementById("tr").style.cssText = "color: green"
    }
    else
    {
      document.getElementById("tr").style.cssText = "color: red"
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
}
