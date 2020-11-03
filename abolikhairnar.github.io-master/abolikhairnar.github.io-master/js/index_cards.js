$(document).ready(function () {
  var data = [{
    id: 0,
    title: 'Customer Segmentation:<br>"Citi Ventures"',
    description: "Conducted research to estimate the effect of human capital factors on financial performance of top firms in US to define a new marketing strategy. Implemented the applied state-of-art Causal Inference techniques and derived business insights to solve real world problems. Identified key revenue driving human capital factors in developing a new strategic model that uses “knowledge-based capital” as the basis to identify clients and industries positioned for sustainable long-term growth. Demonstrated strong presentation skills by reporting the key business insights derived from complex models to the board of executives<br><b>Tools: Python, MySQL.</b>",
    demoUrl: "https://www.citi.com/ventures/",
    previewUrl: "img/segmentation.png",
    category: "data"
  }, {
    id: 1,
    title: "Time Series Analysis:<br>Flood Frequency Analysis for ungauged catchments",
    description: "Extracted and analyzed raw flood data for Tapi basin in India using tools like Python and R programming language. Designed predictive models to forecast flood risk for ungauged catchments by applying algorithms such as Neural Network, Random Forest, XGBoost, and Time Series methods to design and ensemble the models<br><b>Tools: Python, Scikit-Learn, XGBoost, Github.</b>",
    demoUrl: "https://drive.google.com/file/d/1bQf5edAPvu2lskw7HRFhYDKw57OQFlh2/view?usp=sharing",
    previewUrl: "img/flood.jpg",
    category: "data"
  }, {
    id: 2,
    title: "Diseased tree classifier:<br>using satellite imagery data",
    description: "Predicted wilted trees with accuracy of 82 % using supervised machine learning algorithms like Decision Trees, Support Vector Machines and K-Nearest Neighbors.<br><b>Tools: Python, Jupyter Notebook.</b>",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/Diseased_trees_classifier",
    previewUrl: "img/wilted.jpg",
    category: "data"
  }, {
    id: 3,
    title: "Machine Learning:<br>Implementing gradient descent from scratch in Python",
    description: "<br>Gradient descent is one of the most important concepts in machine learning, it is the heart of many techniques which gives machines the power to optimize current solutions - to literally “learn” to do things better. Objective of this repo is to estimate the best fitting linear model by writing my own gradient descent function from scratch.<b> Tools: Python.</b>",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/Gradient%20descent",
    previewUrl: "img/gd.jpeg",
    category: "data2"
  }, {    
    id: 7,
    title: "Survival prediction for Titanic",
    description: "Built a predictive model to identify: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).<br><b>Tools: Matplotlib, Seaborn.</b>",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/Titanic",
    previewUrl: "img/titanic.jpg",
    category: "data2"
  }, {
    id: 4,
    title: "Linear Regression:<br>House Price Prediction",
    description: "Constructed exploratory modeling and predictive modeling to predict new house prices.<br><b>Tools: Regression Models (OLS, Ridge, Lasso, Elastic Net).</b>",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/Predict_housing_prices",
    previewUrl: "img/house.jpg",    
    category: "data2"  
  }, {
    id: 6,
    title: "Topic Modeling for United Nations (UN) speeches",
    description: "Building an LDA model to identify the key topics in the UN speeches.<br><b>Tools: R, R markdown, ggplot, NLP.</b>",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/NLP%20Projects/UN%20speeches%20analysis",
    previewUrl: "img/un.png",
    category: "data3"
  }, {
    id: 5,
    title: "Amazon Reviews Sentiment Analysis",
    description: "Constructed and tuned a model capable of predicting sentiments for unlabeled reviews in R<br><b>Tools: R, NLP, Bag-of-Words, R markdown.",
    demoUrl: "https://github.com/agkhairnar/Projects/tree/master/NLP%20Projects/Amazon%20reviews%20analysis",
    previewUrl: "img/amazon.png",
    category: "data3"
  }];

  var frontend1 = data.filter(function (item) {
    return item.category === 'data';
  });

  var frontend2 = data.filter(function (item) {
    return item.category === 'data2';
  });

  var frontend3 = data.filter(function (item) {
    return item.category === 'data3';
  });

  var frontend5 = data.filter(function (item) {
    return item.category === 'data5';
  });


  function removeActiveClass() {
    $('.center').find('a').each(function (i, el) {
      if ($(el).hasClass('active')) {
        $(el).removeClass('active');
      }
    });
  }
  function removeOpenClass() {
    $('.container').each(function (i, el) {
      if ($(el).hasClass('open')) {
        $(el).removeClass('open');
      }
    });
  }


  $('.js-open-works').on('click', function (e) {
    e.preventDefault();
    removeOpenClass();
    $('.works').addClass('open');
    $('.close-arrow').show(600);
    $('.center').animate({
      opacity: '1'
    }, 500);
    removeActiveClass();
    $('.center').find('.js-open-works').addClass('active');
  });



  function generateCard(data) {
    var title = data.title;
    var desc = data.description;
    var imgUrl = data.previewUrl;
    var demoUrl = data.demoUrl;
    return '<li class="card animated zoomIn"><div class="card-media"><img src=' + imgUrl + ' alt="#" /></div><div class="card-content"><h4>' + title + '</h4><p>' + desc + '</p></div><div class="card-actions"><a target="_blank" href=' + demoUrl + ' class="view-demo"><b>View Detail</b></a></div></li>';
  }

  function renderToDom(data) {
    data.forEach(function (item) {
      var card = generateCard(item);
      //render card to dom
      //console.log(card);
      $('.js-list').append(card);
    });
  }
  
  function renderToDom2(data) {
    data.forEach(function (item) {
      var card = generateCard(item);
      //render card to dom
      //console.log(card);
      $('.js-list2').append(card);
    });
  }
  
  function renderToDom3(data) {
    data.forEach(function (item) {
      var card = generateCard(item);
      //render card to dom
      //console.log(card);
      $('.js-list3').append(card);
    });
  }

  function renderToDom5(data) {
    data.forEach(function (item) {
      var card = generateCard(item);
      //render card to dom
      //console.log(card);
      $('.js-list5').append(card);
    });
  }


  //for intial rendering
  renderToDom(frontend1);
  renderToDom2(frontend2);
  renderToDom3(frontend3);
  renderToDom5(frontend5);


  function clearDom() {
    $('.js-list').empty();
  }
  $('input[type=radio]').on('change', function (e) {
    switch (e.target.value) {
      case 'frontend':
        clearDom();
        renderToDom(frontend);
        break;
      case 'backend':
        clearDom();
        renderToDom(backend);
        break;
      case 'react':
        clearDom();
        renderToDom(react);
        break;
      case 'meteor':
        clearDom();
        renderToDom(meteor);
        break;
    }
  });
});

if (document.createElement("p").style.flex === undefined) {
  alert("no flexbox support");
}

$.backstretch(["https://res.cloudinary.com/vinaypuppal/image/upload/c_scale,w_1024/v1464896760/photo-1428223501723-d821c5d00ca3_knl5ki.jpg", "https://res.cloudinary.com/vinaypuppal/image/upload/c_scale,w_1024/v1464896839/photo-1432888498266-38ffec3eaf0a_ukva1r.jpg"], {
  duration: 20000,
  fade: 'slow'
});