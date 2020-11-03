$(document).ready(function(){
  
  
  /*SMOOTH SCROLLING
  -------------------*/
  
  $('a').click(function(){
    $('html, body').animate({
        scrollTop: $( $.attr(this, 'href') ).offset().top
    }, 500);
    return false;
});
  
/*END SMOOTH SCROLLING*/
  

  /*Animation TITLE Page*/
  $('.title-intro').addClass('animated bounceInDown');
  
   $(document).foundation();
  
  /*set the height of the home page section according size
  iewport of the visitor. Set at the same time the position of the title of this section
  and the position of the social media icon if presents.
  */
  var heightBrowser = window.innerHeight;
  $('.about-me').css('min-height',heightBrowser);
  
  $('.title-intro h1').css('margin-top',(heightBrowser/2)-130);
  
  if($('.social').css('display') !== 'none'){
    $('.social-media').css('margin-top',(heightBrowser/2)-130);
 
  }
  
  /*display SOCIAL MEDIA
  --------------------------*/
  var dist;
  
  $(this).scroll(function(){
    dist = $(this).scrollTop();
    console.log(dist);
    if(dist > heightBrowser-220){
      console.log('on affiche');
      $('.social-media').css('display','block').addClass('animated bounceIn');
    }else{
      $('.social-media').css('display','none').removeClass('animated bounceIn');
    }
  });
  
  

  
  /*PORTFOLIO
  --------------*/
  
  $('.more-info-btn').click(function(){
    
    $(this).next().slideToggle();
    $(this).toggleClass("fa-minus");
  
  });

});