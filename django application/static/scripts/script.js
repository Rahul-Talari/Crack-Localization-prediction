console.log('Script injected!!!') ; 

$(document).ready(function () {
    console.log('Document is ready.'); // Debug message 1
  
    // Get the elements
    var outputCard = $('.rectangle-group');
    var predictButton = $('#predict-button');
    var loader = $('.loader');
  
    console.log('Elements selected:', outputCard, predictButton, loader); // Debug message 2
  
    // Add click event listener to the Predict button
    predictButton.on('click', function () {
      console.log('Predict button clicked.'); // Debug message 3
  
      // Show the loader animation
      loader.show();
  
      console.log('Loader shown.'); // Debug message 4
  
      // Simulate a delay of 2 seconds (replace with actual prediction process)
      setTimeout(function () {
        // Hide the loader after 2 seconds
        loader.hide();
  
        console.log('Loader hidden.'); // Debug message 5
  
        // Remove the blur from the output card
        outputCard.removeClass('blur');
  
        console.log('Blur removed from output card.'); // Debug message 6
      }, 50);

    });
  });
  

// try{    // Get the elements
//     const outputCard = document.querySelector('.rectangle-group');
//     const predictButton = document.querySelector('.predict');
//     const loader = document.querySelector('.loader');

//     // Add click event listener to the Predict button
//     predictButton.addEventListener('click', () => {
//     // Show the loader animation
//     loader.style.display = 'block';

//     // Simulate a delay of 2 seconds (replace with actual prediction process)
//     setTimeout(() => {
//         // Hide the loader after 2 seconds
//         loader.style.display = 'none';

//         // Remove the blur from the output card
//         outputCard.classList.remove('blur');
//     }, 2000);
//     });
// }
// catch(error)
// {
//     console.log(error.message) ; 
//     console.error('An error occurred:', error.message);
// }




// const predictButton = document.getElementById('predict-button');

// predictButton.addEventListener('click', () => {
//   console.log('Button clicked!'); // Debug message
//   try {
//     // Your code that may cause an error
//     const outputCard = document.querySelector('.rectangle-group');

//     if (!outputCard) {
//       console.error('.rectangle-group element not found!');
//       return;
//     }

//     // Show the loader animation
//     const loader = document.querySelector('.loader');

//     if (!loader) {
//       console.error('.loader element not found!');
//       return;
//     }

//     loader.style.display = 'block';

//     // Simulate a delay of 2 seconds (replace with actual prediction process)
//     setTimeout(() => {
//       // Hide the loader after 2 seconds
//       loader.style.display = 'none';

//       // Remove the blur from the output card
//       outputCard.classList.remove('blur');
//     }, 2000);
//   } catch (error) {
//     // Handle the error gracefully
//     console.error('An error occurred:', error.message);
//   }
// });
