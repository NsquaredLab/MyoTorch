.. |logo| image:: _static/myotorch_logo.png
   :height: 80px
   :align: middle

Welcome to |logo|
===========================

**The AI toolkit for myocontrol research**

MyoTorch is your cutting-edge **research** companion for unlocking the secrets hidden within biomechanical data! It's specifically designed for exploring the complex interplay between **electromyography (EMG)** signals, **kinematics** (movement), and **kinetics** (forces).

Leveraging the power of **PyTorch** and **PyTorch Lightning**, MyoTorch provides a comprehensive suite of tools for researchers and developers working with myoelectric signal analysis and AI-driven biomechanical applications.

.. raw:: html

    <div>
        <form class="bd-search align-items-center" action="search.html" method="get">
          <input type="search" class="form-control search-front-page" name="q" id="search-input" placeholder="&#128269; Search the docs ..." aria-label="Search the docs ..." autocomplete="off">
        </form>
    </div>

Key Features
-----------

* **Data loaders** and **preprocessing filters** tailored for biomechanical signals
* Peer-reviewed **AI models** and components for analysis and prediction tasks
* Comprehensive visualization tools
* Essential **utilities** to streamline the research workflow

.. important::
   MyoTorch is built for **research**. While powerful, it's evolving and may not have the same level of stability as foundational libraries like NumPy.

Package Structure
----------------

* **myotorch**: Main package containing:
   * **datasets**: Data loaders, dataset creators, and preprocessing filters
   * **models**: AI models and components for training and evaluation
   * **utils**: Support for data handling, model training, and analysis
* **examples**: Practical examples including tutorials and use cases

Research
----------------------

MyoTorch has been used in several publications:

* IEEE Transactions on Biomedical Engineering (10.1109/TBME.2024.3432800)
* Journal of Neural Engineering (10.1088/1741-2552/ad3498)
* IEEE Transactions on Neural Systems and Rehabilitation Engineering (10.1109/TNSRE.2023.3295060)
* And more...

.. raw:: html

    <!-- Raw HTML for Carousel -->
    <style>
    .paper-carousel-container {
        width: 95%;
        max-width: 80em; /* Keep max-width */
        /* Remove fixed height, let content define it */
        overflow: hidden;
        margin: 2em auto; /* Use relative margin */
        border: 0px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .paper-carousel-track {
        display: flex;
        /* Width = (Number of items / Items visible) * 100% = (12 / 2) * 100% */
        width: 600%; /* 6 items wide relative to container (fits 2 visible) */
        animation: scroll 60s linear infinite; /* Keep animation */
    }

    .paper-carousel-track a {
        display: flex; /* Change to flex */
        align-items: center; /* Vertically center */
        justify-content: center; /* Horizontally center */
        /* Width relative to track width. 12 items total, so each is 1/12th of track */
        /* (1/12) * 400% = 33.33% of container width */
        width: calc(100% / 12);
        margin-right: 1em; /* Relative margin */
        flex-shrink: 0; /* Prevent shrinking */
    }
    
    .paper-carousel-track img {
        width: auto; /* Let height control width via aspect ratio */
        max-width: 100%; /* Scale image within its 'a' container */
        height: auto; /* Maintain aspect ratio */
        /* Remove fixed max-height */
        object-fit: contain;
        display: block;
        border: 1px solid #eee; /* Optional subtle border */
    }
    
    @keyframes scroll {
        0% {
            transform: translateX(0);
        }
        100% {
            /* Translate by half the track width (since content is duplicated) */
            transform: translateX(-50%);
        }
    }
    
    .paper-carousel-container:hover .paper-carousel-track {
        animation-play-state: paused;
    }

    /* Media Query for smaller screens */
    @media (max-width: 768px) {
        .paper-carousel-container {
            margin: 1em auto; /* Smaller margin */
        }
        .paper-carousel-track a {
            margin-right: 0.5em; /* Smaller margin */
        }
        /* Optional: Adjust animation speed or other properties */
         .paper-carousel-track {
            /* Example: Slow down animation slightly on mobile */
            /* animation-duration: 70s; */
        }
    }
    </style>
    
    <div class="paper-carousel-container">
        <div class="paper-carousel-track">
            <!-- List images twice, wrapped in links -->
            <a href="https://doi.org/10.1109/EMBC48229.2022.9870937" target="_blank"><img src="_static/papers/Accurate.jpg" alt="Accurate Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2023.3295060" target="_blank"><img src="_static/papers/Proportional.jpg" alt="Proportional Paper"></a> 
            <a href="https://doi.org/10.1088/1741-2552/ad3498" target="_blank"><img src="_static/papers/Influence.jpg" alt="Influence Paper"></a>
            <a href="http://www.iadisportal.org/ijcsis/papers/2024190101.pdf" target="_blank"><img src="_static/papers/Analysis.jpg" alt="Analysis Paper"></a>          
            <a href="https://doi.org/10.1109/TBME.2024.3432800" target="_blank"><img src="_static/papers/Learning.jpg" alt="Learning Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2024.3472063" target="_blank"><img src="_static/papers/Identification.jpg" alt="Identification Paper"></a>
            <!-- Add a new row to create a continuous scroll -->
            <a href="https://doi.org/10.1109/EMBC48229.2022.9870937" target="_blank"><img src="_static/papers/Accurate.jpg" alt="Accurate Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2023.3295060" target="_blank"><img src="_static/papers/Proportional.jpg" alt="Proportional Paper"></a> 
            <a href="https://doi.org/10.1088/1741-2552/ad3498" target="_blank"><img src="_static/papers/Influence.jpg" alt="Influence Paper"></a>
            <a href="http://www.iadisportal.org/ijcsis/papers/2024190101.pdf" target="_blank"><img src="_static/papers/Analysis.jpg" alt="Analysis Paper"></a>          
            <a href="https://doi.org/10.1109/TBME.2024.3432800" target="_blank"><img src="_static/papers/Learning.jpg" alt="Learning Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2024.3472063" target="_blank"><img src="_static/papers/Identification.jpg" alt="Identification Paper"></a>
             
        </div>
    </div>
    <!-- End Raw HTML -->

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   auto_examples/index.rst
   api_documentation.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Development:
   :hidden:

   contributing.rst
   Changelog <../CHANGELOG.md>
