<picture align="center">
  <img alt="CSS Logo" src="resources/header.png">
</picture>

-----------------

# Customer Success Scorecard (CSS) Mini

Customer Success Scorecard (CSS) helps our customers better understand their use of Salesforce products. Whether it is implementation details or license activations, you can track progress over time and double-click on specific areas of your organization’s performance.

As you can imagine, this takes a lot of data: telemetry, business metrics, CRM data, and more. Data Cloud lets us bring all of this data into one place where we can run machine learning models to give meaning to this data.

One question we get all the time from customers is, “How do we build our own CSS?” While we have been happy to share our journey, this is the first time we are showing everything you need in step-by-step detail to set up a miniature version of CSS from scratch!

### What is this repo?
This repository comprises the machine learning back-end that is leveraged via Einstein Studio's _[bring your own model](https://help.salesforce.com/s/articleView?language=en_US&id=sf.c360_a_bring_your_own_model.htm&type=5)_ functionality.

There are three elements within this repo:
1. **`css-mini` codebase** — This is the Python package that defines our scoring algorithm. You can install it via `pip` and use it for anything you would like.
1. **`infrastructure`** — Terraform code to deploy a minimal ML service within AWS.
1. **`example/`** — Here we host tutorials involving CSS. If you find yourself here as part of a blog or tutorial, that is the right place to start.

### Python Package Installation
Packaged in this repo is our Python logic that defines a scoring algorithm to score arbitrary numerical values then group them in a hierarchy for high level insights.

You can install the package directly from `pip`.
```{bash}
pip install css-mini
```

### Issues?
Visit our Issues page to submit any questions, concerns, or bugs!
